"""Model training, leak-free walk-forward CV, shrinkage, and prediction (v3.0).

Key v3.0 upgrades:
1. Leak-free `sklearn.pipeline.Pipeline(SimpleImputer, StandardScaler, model)`
   used identically inside CV, conformal calibration, and production fits.
2. Capacity right-sizing for low-SNR agricultural prices:
   - 1d–7d: Regularized Huber + Ridge linear blend (no slow macro/weather noise)
   - 14d & 28d: Ridge + shallow Huber-loss GBR (`max_depth=2`, `min_samples_leaf>=15`)
     with CV-learned blend weights and exact ensemble CV evaluation
   - 90d: Scaled BayesianRidge on stationary fundamentals + recency weights
   - Regime (6m bear risk): L2-regularized LogisticRegression (calibrated probs)
3. Exponential recency sample weights so structural regime shifts (2023–26
   Guatemala supply shock) adapt quickly rather than being dragged down by
   decade-old low-price regimes.
4. Explicit out-of-sample shrinkage `lambda_h in [0, 1]` toward the random walk
   (`r_final = lambda_h * r_raw`), preventing negative skill on near-martingale
   horizons while preserving genuine signal where it exists.
5. Forecasts log-returns on the causal house-mix-adjusted index (`adj_price`),
   reconstructed cleanly to price levels, with term-structure coherence smoothing
   and volatility-scaled Online Adaptive Conformal Inference (ACI) intervals.
"""

import logging
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, HuberRegressor, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    CONFORMAL_ALPHA,
    CONFORMAL_VOL_FLOOR,
    CONFORMAL_WINDOW_DAILY,
    MAX_SHORT_HORIZON_DEVIATION,
    MODEL_VERSION,
    MODELS_DIR,
    RECENCY_HALFLIFE_DAYS,
    TARGET_MATCH_TOLERANCE_DAYS,
    WF_CONFIG,
    ensure_dirs,
)
from pipeline.conformal import calibrate, resolve_return_band, walk_forward_residuals
from pipeline.db import get_conn

log = logging.getLogger(__name__)


# ── Compact, stationary, horizon-matched feature sets ────────────────────
# Zero raw INR price levels; zero slow macro/weather noise at 1d–7d.

FEATS_SHORT = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5",
    "dev_ma_1", "dev_ma_2", "dev_ma_3", "dev_ma_4",
    "rvol_short", "rvol_long", "rvol_ratio", "zscore_ma3",
    "range_pos", "rsi", "boll_pos",
    "vol_ratio", "sell_through", "max_avg_spread", "house_effect_log",
    "disp_cv_now", "disp_cv_ma", "unsold_now", "unsold_ma", "grade_spread_ma",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]

FEATS_14D = FEATS_SHORT + [
    "week_sin", "week_cos",
    "wedding_season", "harvest_season", "peak_harvest",
    "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    "is_lean", "is_strong",
    "rain_anomaly", "fcst_rain_14d_anom",
]

FEATS_28D = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5",
    "dev_ma_1", "dev_ma_2", "dev_ma_3", "dev_ma_4",
    "rvol_short", "rvol_long", "zscore_ma3", "range_pos", "rsi",
    "vol_ratio", "sell_through", "max_avg_spread",
    "disp_cv_ma", "unsold_ma", "grade_spread_ma",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "harvest_season", "peak_harvest", "pre_diwali_period", "pre_eid_period",
    "is_lean", "is_strong",
    "rain_cum_1", "rain_cum_2", "rain_cum_3", "rain_anomaly",
    "gt_rain_cum_1", "gt_rain_cum_2",
    "enso", "enso_lag6m", "enso_lag12m",
    "usdinr_pct_short", "usdinr_pct_long", "crudeoil_pct_short",
    "gt_qty_3m_log", "gt_qty_yoy", "sa_qty_yoy",
]

FEATS_90D = [
    "ret_1", "ret_2", "ret_3",
    "dev_ma_1", "dev_ma_2", "dev_ma_3",
    "rvol_long", "range_pos", "vol_ratio",
    "month_sin", "month_cos", "is_lean", "is_strong",
    "rain_cum_3", "rain_anomaly", "gt_rain_cum_3",
    "enso", "enso_lag6m", "enso_lag12m",
    "gt_qty_3m_log", "gt_qty_yoy", "sa_qty_yoy",
    "log_price_to_cost", "vol_vs_2yr",
]

FEATS_REGIME = [
    "ret_2", "ret_3", "ret_4",
    "dev_ma_2", "dev_ma_3", "rvol_long",
    "month_sin", "month_cos", "is_lean", "is_strong",
    "rain_cum_3", "rain_anomaly",
    "enso", "enso_lag6m", "enso_lag12m",
    "gt_qty_3m_log", "gt_qty_yoy", "gt_unit_price_log", "sa_qty_yoy",
    "log_price_to_cost", "cycle_age_norm", "gt_plantation_ma12", "vol_vs_2yr",
]


def _available_feats(df: pd.DataFrame, feat_list: list[str], min_non_null: int = 20) -> list[str]:
    """Return features that exist in `df` and have enough non-null values to be useful."""
    out = []
    for f in feat_list:
        if f in df.columns and int(df[f].notna().sum()) >= min_non_null:
            out.append(f)
    return out


def _price_col(df: pd.DataFrame) -> str:
    """Use the causal house-mix-adjusted index when available, else raw avg_price."""
    if "adj_price" in df.columns and df["adj_price"].notna().sum() > 0:
        return "adj_price"
    return "avg_price"


def _recency_weights(dates, halflife_days: float) -> np.ndarray:
    """Compute normalized exponential time-decay weights ending at the latest date."""
    n = len(dates)
    if n == 0 or not halflife_days or halflife_days <= 0:
        return np.ones(n, dtype=float)
    # Callers pass numpy arrays, which pd.to_datetime turns into a
    # DatetimeIndex; TimedeltaIndex has no .dt accessor, so divide by a
    # one-day Timedelta instead (works for Index and Series alike).
    ts = pd.DatetimeIndex(pd.to_datetime(dates))
    age_days = np.asarray((ts.max() - ts) / pd.Timedelta(days=1), dtype=float)
    age_days = np.clip(age_days, 0.0, None)
    w = np.exp(-np.log(2.0) * age_days / float(halflife_days))
    mean_w = float(np.mean(w))
    return w / mean_w if mean_w > 0 else np.ones(n, dtype=float)


# ── Targets ──────────────────────────────────────────────────────────────

def make_return_target(
    df: pd.DataFrame,
    horizon: int,
    price_col: str | None = None,
) -> pd.Series:
    """Calendar-period log-return target: log(price[t+h] / price[t])."""
    col = price_col or _price_col(df)
    p = pd.to_numeric(df[col], errors="coerce").clip(lower=1.0)
    return np.log(p.shift(-horizon) / p)


def make_return_target_calendar(
    df: pd.DataFrame,
    horizon_days: int,
    price_col: str | None = None,
    date_col: str = "date",
    tolerance_days: int = TARGET_MATCH_TOLERANCE_DAYS,
) -> pd.Series:
    """Log-return target aligned to CALENDAR days on the auction-indexed frame.

    Matches the exact settlement rule in `validate.py`: first auction at or
    after `date + horizon_days` within `tolerance_days`.
    """
    col = price_col or _price_col(df)
    dates = pd.to_datetime(df[date_col])
    current = pd.to_numeric(df[col], errors="coerce")

    priced = pd.DataFrame({"_key": dates, "_future": current}).dropna(
        subset=["_future"]
    ).sort_values("_key")
    want = pd.DataFrame({
        "_key": dates + pd.Timedelta(days=horizon_days),
        "_pos": np.arange(len(df)),
    }).sort_values("_key")

    matched = pd.merge_asof(
        want, priced, on="_key", direction="forward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    future = pd.Series(np.nan, index=df.index, dtype=float)
    future.iloc[matched["_pos"].to_numpy()] = matched["_future"].to_numpy()
    return np.log(future.clip(lower=1.0) / current.clip(lower=1.0))


# ── Leak-free Model Factories (all wrapped in Pipeline) ──────────────────

def _make_pipe(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _huber_short(horizon: int) -> Pipeline:
    # Stronger regularization at very short horizons where SNR is lowest
    alpha = 25.0 if horizon <= 3 else 15.0
    return _make_pipe(HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=500))


def _ridge_short(horizon: int) -> Pipeline:
    alpha = 150.0 if horizon <= 3 else 80.0
    return _make_pipe(Ridge(alpha=alpha))


def _gbr_shallow(min_leaf: int = 30, n_estimators: int = 60, lr: float = 0.04) -> Pipeline:
    """Shallow Huber-loss GBR sized for low-SNR commodity returns."""
    return _make_pipe(GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=lr,
        subsample=0.75,
        min_samples_leaf=min_leaf,
        random_state=42,
    ))


def _ridge_med(alpha: float = 40.0) -> Pipeline:
    return _make_pipe(Ridge(alpha=alpha))


def _bayesian_90d() -> Pipeline:
    return _make_pipe(BayesianRidge(
        alpha_1=1e-4, alpha_2=1e-4,
        lambda_1=1e-3, lambda_2=1e-3,
    ))


def _logreg_regime() -> Pipeline:
    """L2-regularized LogisticRegression for calibrated bear-regime probabilities."""
    return _make_pipe(LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=500,
        random_state=42,
    ))


# ── Fit helper (supports optional recency sample_weight cleanly) ─────────

def _fit_pipe(pipe: Pipeline, X: np.ndarray, y: np.ndarray, sample_weight=None) -> Pipeline:
    if sample_weight is not None:
        try:
            pipe.fit(X, y, model__sample_weight=sample_weight)
            return pipe
        except TypeError:
            pass
    pipe.fit(X, y)
    return pipe


# ── Ensemble Predictor Wrappers ──────────────────────────────────────────

class BlendedReturnModel:
    """Two-component regularized blend with explicit random-walk shrinkage `lam`."""

    def __init__(self, m1: Pipeline, m2: Pipeline | None = None, w1: float = 0.5, lam: float = 1.0):
        self.m1 = m1
        self.m2 = m2
        self.w1 = float(w1)
        self.lam = float(lam)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        p1 = self.m1.predict(X)
        if self.m2 is None:
            return p1
        p2 = self.m2.predict(X)
        return self.w1 * p1 + (1.0 - self.w1) * p2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lam * self.predict_raw(X)


def _optimize_blend_and_shrinkage(
    p1_oos: np.ndarray,
    p2_oos: np.ndarray | None,
    y_oos: np.ndarray,
    min_gain: float = 0.002,
) -> tuple[float, float]:
    """Find OOS-optimal blend weight `w1` and shrinkage `lam in [0, 1]` minimizing MAE.

    If even the best shrinkage fails to beat predicting 0 (random walk) by at
    least `min_gain` relative MAE, `lam` is collapsed to 0.0 (pure persistence),
    preventing high-variance horizons from destroying forecast skill.
    """
    y = np.asarray(y_oos, dtype=float)
    p1 = np.asarray(p1_oos, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p1)
    if p2_oos is not None:
        p2 = np.asarray(p2_oos, dtype=float)
        mask &= np.isfinite(p2)
    else:
        p2 = None

    if mask.sum() < 20:
        return (0.5 if p2 is not None else 1.0), 0.25

    y = y[mask]
    p1 = p1[mask]
    p2 = p2[mask] if p2 is not None else None

    naive_mae = float(np.mean(np.abs(y)))
    if naive_mae <= 1e-12:
        return (0.5 if p2 is not None else 1.0), 0.0

    w_grid = np.linspace(0.0, 1.0, 11) if p2 is not None else [1.0]
    lam_grid = np.linspace(0.0, 1.0, 41)

    best_mae = naive_mae
    best_w = 0.5 if p2 is not None else 1.0
    best_lam = 0.0

    for w in w_grid:
        raw = p1 if p2 is None else (w * p1 + (1.0 - w) * p2)
        for lam in lam_grid:
            mae = float(np.mean(np.abs(y - lam * raw)))
            if mae < best_mae:
                best_mae = mae
                best_w = float(w)
                best_lam = float(lam)

    # Require a tiny margin over persistence before accepting non-zero shrinkage
    rel_gain = (naive_mae - best_mae) / naive_mae
    if rel_gain < min_gain:
        return best_w, 0.0

    return best_w, best_lam


# ── Walk-Forward Cross-Validation (with Nested Shrinkage) ────────────────

def walk_forward_cv_ensemble(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    fn1,
    fn2,
    min_train: int,
    step: int,
    eval_win: int,
    purge: int = 0,
    anchor_col: str = "avg_price",
    halflife_days: float | None = None,
) -> tuple[dict, float, float]:
    """Run strictly leak-free walk-forward CV with nested blend & shrinkage.

    At each fold `k`, `w1` and `lam` are estimated ONLY on prior out-of-sample
    folds `< k` (nested CV), so reported `skill` and `theil_u` are 100% honest
    out-of-sample metrics. Also returns `(final_w1, final_lam)` fit on all OOS
    folds for production deployment.
    """
    cols = [target_col, anchor_col, "date"] + features
    df_clean = df.dropna(subset=[target_col, anchor_col]).copy()
    X = df_clean[features].values
    y = df_clean[target_col].values
    anchors = df_clean[anchor_col].values
    dates = df_clean["date"].values
    n = len(df_clean)

    oos_p1, oos_p2, oos_y = [], [], []
    all_preds, all_actuals, all_naive = [], [], []
    fold = 0

    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[i: i + eval_win], y[i: i + eval_win]
        d_train = dates[:train_end]

        if len(X_train) < 20 or len(X_test) == 0:
            i += step
            continue

        sw = _recency_weights(d_train, halflife_days) if halflife_days else None

        m1 = _fit_pipe(fn1(), X_train, y_train, sample_weight=sw)
        p1 = m1.predict(X_test)

        if fn2 is not None:
            m2 = _fit_pipe(fn2(), X_train, y_train, sample_weight=sw)
            p2 = m2.predict(X_test)
        else:
            p2 = None

        # Nested shrinkage: determine (w1, lam) strictly from earlier OOS folds
        if len(oos_y) >= 30:
            w_fold, lam_fold = _optimize_blend_and_shrinkage(
                np.array(oos_p1),
                np.array(oos_p2) if fn2 is not None else None,
                np.array(oos_y),
            )
        else:
            w_fold = 0.5 if fn2 is not None else 1.0
            lam_fold = 0.25

        raw_fold = p1 if p2 is None else (w_fold * p1 + (1.0 - w_fold) * p2)
        pred_ret = lam_fold * raw_fold

        a = anchors[i: i + eval_win]
        all_preds.extend(a * np.exp(pred_ret))
        all_actuals.extend(a * np.exp(y_test))
        all_naive.extend(a)

        oos_p1.extend(p1)
        if p2 is not None:
            oos_p2.extend(p2)
        oos_y.extend(y_test)

        fold += 1
        i += step

    final_w, final_lam = _optimize_blend_and_shrinkage(
        np.array(oos_p1),
        np.array(oos_p2) if fn2 is not None else None,
        np.array(oos_y),
    )

    if not all_preds:
        return (
            {"mape": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
             "folds": 0, "naive_mape": np.nan, "naive_mae": np.nan,
             "skill": np.nan, "theil_u": np.nan, "dir_acc": np.nan},
            final_w,
            final_lam,
        )

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    naive = np.array(all_naive)

    errors = np.abs(preds - actuals)
    pct_errors = errors / np.abs(actuals).clip(min=1.0)
    n_err = np.abs(naive - actuals)

    mape = float(np.mean(pct_errors))
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    naive_mae = float(np.mean(n_err))
    naive_rmse = float(np.sqrt(np.mean(n_err ** 2)))
    naive_mape = float(np.mean(n_err / np.abs(actuals).clip(min=1.0)))

    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))

    pred_dir = np.sign(preds - naive)
    act_dir = np.sign(actuals - naive)
    dir_acc = float(np.mean(pred_dir == act_dir))

    metrics = {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "folds": fold,
        "naive_mape": naive_mape,
        "naive_mae": naive_mae,
        "skill": float(1.0 - mae / naive_mae) if naive_mae > 0 else np.nan,
        "theil_u": float(rmse / naive_rmse) if naive_rmse > 0 else np.nan,
        "dir_acc": dir_acc,
        "blend_w1": round(final_w, 3),
        "shrinkage_lam": round(final_lam, 3),
    }
    return metrics, final_w, final_lam


def walk_forward_cv(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    model_fn,
    min_train: int,
    step: int,
    eval_win: int,
    purge: int = 0,
    anchor_col: str | None = None,
) -> dict:
    """Backward-compatible single-model walk-forward CV wrapper."""
    a_col = anchor_col or _price_col(df)
    metrics, _, _ = walk_forward_cv_ensemble(
        df, features, target_col,
        fn1=model_fn, fn2=None,
        min_train=min_train, step=step, eval_win=eval_win,
        purge=purge, anchor_col=a_col, halflife_days=None,
    )
    return metrics


def walk_forward_auc(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    model_fn,
    min_train: int,
    step: int,
    eval_win: int,
    purge: int = 0,
    halflife_days: float | None = None,
) -> dict:
    """Walk-forward CV for binary classifiers (leak-free via Pipeline)."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[features].values
    y = df_clean[target_col].values
    dates = df_clean["date"].values if "date" in df_clean.columns else None
    n = len(df_clean)

    probs, actuals = [], []
    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        if train_end >= 20 and len(np.unique(y[:train_end])) > 1:
            sw = (
                _recency_weights(dates[:train_end], halflife_days)
                if (dates is not None and halflife_days) else None
            )
            model = _fit_pipe(model_fn(), X[:train_end], y[:train_end], sample_weight=sw)
            probs.extend(model.predict_proba(X[i: i + eval_win])[:, 1])
            actuals.extend(y[i: i + eval_win])
        i += step

    if len(set(actuals)) < 2:
        return {"auc": np.nan, "brier": np.nan, "n": len(actuals)}
    return {
        "auc": float(roc_auc_score(actuals, probs)),
        "brier": float(brier_score_loss(actuals, probs)),
        "n": len(actuals),
    }


# ── Fit-predict closures for volatility-scaled conformal calibration ─────

def _fp_ensemble(fn1, fn2, w1: float, lam: float, halflife_days: float | None):
    """Exact replica of a deployed BlendedReturnModel for conformal residual collection."""
    def fp(X_train, y_train, X_test, dates_train=None):
        sw = (
            _recency_weights(dates_train, halflife_days)
            if (dates_train is not None and halflife_days) else None
        )
        m1 = _fit_pipe(fn1(), X_train, y_train, sample_weight=sw)
        p1 = m1.predict(X_test)
        if fn2 is None:
            return lam * p1
        m2 = _fit_pipe(fn2(), X_train, y_train, sample_weight=sw)
        p2 = m2.predict(X_test)
        return lam * (w1 * p1 + (1.0 - w1) * p2)
    return fp


# ── Train all models ─────────────────────────────────────────────────────

def _log_cv(name: str, cv: dict) -> None:
    """Log a CV result alongside the naive baseline, blend weight, and shrinkage."""
    skill, theil = cv.get("skill"), cv.get("theil_u")
    base = f"{name} WF-CV: MAPE={cv['mape']:.3f}, MAE={cv['mae']:.1f}"
    if skill is None or np.isnan(skill):
        log.info(base)
        return
    verdict = "BEATS naive" if skill > 0 else ("MATCHES naive" if skill >= -0.005 else "WORSE than naive")
    lam = cv.get("shrinkage_lam", 1.0)
    log.info(
        f"{base}, naive MAPE={cv['naive_mape']:.3f} MAE={cv['naive_mae']:.1f} | "
        f"skill={skill:+.3f} TheilU={theil:.3f} dir={cv.get('dir_acc', float('nan')):.1%} "
        f"lam={lam:.2f} -> {verdict}"
    )


def train_all(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
) -> dict:
    """Train all v3.0 models on full data and save to disk."""
    ensure_dirs()
    results = {}
    selected = {}
    shrinkage_map = {}
    blend_map = {}
    conformal = {}

    pcol_d = _price_col(daily)
    pcol_w = _price_col(weekly)
    pcol_m = _price_col(monthly)

    def _calibrate_horizon(key, df, features, target_col, fp, wf_key, purge, window):
        resid, vols = walk_forward_residuals(
            df, features, target_col, fp,
            **WF_CONFIG[wf_key], purge=purge,
            vol_col="rvol_long", date_col="date",
        )
        rec = calibrate(
            resid, vols=vols, alpha=CONFORMAL_ALPHA, window=window,
            min_cal=100 if wf_key == "daily" else 40,
        )
        if rec is None:
            log.warning(f"{key}: too few residuals to calibrate conformal band")
            return
        conformal[key] = rec
        log.info(
            f"{key} conformal (vol-scaled ACI): coverage={rec['coverage']:.2f} "
            f"(target {1 - CONFORMAL_ALPHA:.2f}), width≈{rec['mean_width_pct']:.1f}% of price, "
            f"alpha_t={rec['alpha_t']:.3f}, n={rec['n_resid']}"
        )

    # ── 1d–7d daily models (Huber + Ridge blend on stationary + micro features) ──
    feats_short = _available_feats(daily, FEATS_SHORT, min_non_null=100)
    selected["short"] = feats_short
    hl_daily = RECENCY_HALFLIFE_DAYS["daily"]

    for h in range(1, 8):
        key = f"{h}d"
        target_col = f"target_{key}"
        daily[target_col] = make_return_target_calendar(daily, h, price_col=pcol_d)

        fn1 = lambda _h=h: _huber_short(_h)
        fn2 = lambda _h=h: _ridge_short(_h)
        cv, w1, lam = walk_forward_cv_ensemble(
            daily, feats_short, target_col,
            fn1=fn1, fn2=fn2,
            **WF_CONFIG["daily"], purge=h,
            anchor_col=pcol_d, halflife_days=hl_daily,
        )
        _log_cv(f"{h}-day", cv)
        blend_map[key] = w1
        shrinkage_map[key] = lam

        df_tr = daily.dropna(subset=[target_col])
        sw = _recency_weights(df_tr["date"].values, hl_daily)
        X_tr, y_tr = df_tr[feats_short].values, df_tr[target_col].values
        m1 = _fit_pipe(fn1(), X_tr, y_tr, sample_weight=sw)
        m2 = _fit_pipe(fn2(), X_tr, y_tr, sample_weight=sw)
        model = BlendedReturnModel(m1, m2, w1=w1, lam=lam)

        joblib.dump(model, str(MODELS_DIR / f"model_{key}.pkl"))
        results[key] = {"model": model, "features": feats_short, "cv": cv}

        _calibrate_horizon(
            key, daily, feats_short, target_col,
            _fp_ensemble(fn1, fn2, w1, lam, hl_daily),
            "daily", purge=h, window=CONFORMAL_WINDOW_DAILY,
        )

    # ── 14-day model (Ridge + shallow Huber GBR) ─────────────────────────
    feats_14d = _available_feats(daily, FEATS_14D, min_non_null=100)
    selected["14d"] = feats_14d
    daily["target_14d"] = make_return_target_calendar(daily, 14, price_col=pcol_d)

    fn14_1 = lambda: _ridge_med(alpha=40.0)
    fn14_2 = lambda: _gbr_shallow(min_leaf=30, n_estimators=60, lr=0.04)
    cv_14d, w14, lam14 = walk_forward_cv_ensemble(
        daily, feats_14d, "target_14d",
        fn1=fn14_1, fn2=fn14_2,
        **WF_CONFIG["daily"], purge=14,
        anchor_col=pcol_d, halflife_days=hl_daily,
    )
    _log_cv("14-day", cv_14d)
    blend_map["14d"] = w14
    shrinkage_map["14d"] = lam14

    df_tr = daily.dropna(subset=["target_14d"])
    sw = _recency_weights(df_tr["date"].values, hl_daily)
    X_tr, y_tr = df_tr[feats_14d].values, df_tr["target_14d"].values
    m14 = BlendedReturnModel(
        _fit_pipe(fn14_1(), X_tr, y_tr, sample_weight=sw),
        _fit_pipe(fn14_2(), X_tr, y_tr, sample_weight=sw),
        w1=w14, lam=lam14,
    )
    joblib.dump(m14, str(MODELS_DIR / "model_14d.pkl"))
    results["14d"] = {"model": m14, "features": feats_14d, "cv": cv_14d}

    _calibrate_horizon(
        "14d", daily, feats_14d, "target_14d",
        _fp_ensemble(fn14_1, fn14_2, w14, lam14, hl_daily),
        "daily", purge=14, window=CONFORMAL_WINDOW_DAILY,
    )

    # ── 28-day model (weekly VWAP adj_price, Ridge + shallow Huber GBR) ──
    feats_28d = _available_feats(weekly, FEATS_28D, min_non_null=40)
    selected["28d"] = feats_28d
    weekly["target_28d"] = make_return_target(weekly, 4, price_col=pcol_w)
    hl_weekly = RECENCY_HALFLIFE_DAYS["weekly"]

    fn28_1 = lambda: _ridge_med(alpha=25.0)
    fn28_2 = lambda: _gbr_shallow(min_leaf=15, n_estimators=60, lr=0.04)
    cv_28d, w28, lam28 = walk_forward_cv_ensemble(
        weekly, feats_28d, "target_28d",
        fn1=fn28_1, fn2=fn28_2,
        **WF_CONFIG["weekly"], purge=4,
        anchor_col=pcol_w, halflife_days=hl_weekly,
    )
    _log_cv("28-day", cv_28d)
    blend_map["28d"] = w28
    shrinkage_map["28d"] = lam28

    df_tr = weekly.dropna(subset=["target_28d"])
    if len(df_tr) > 20:
        sw = _recency_weights(df_tr["date"].values, hl_weekly)
        X_tr, y_tr = df_tr[feats_28d].values, df_tr["target_28d"].values
        m28 = BlendedReturnModel(
            _fit_pipe(fn28_1(), X_tr, y_tr, sample_weight=sw),
            _fit_pipe(fn28_2(), X_tr, y_tr, sample_weight=sw),
            w1=w28, lam=lam28,
        )
        joblib.dump(m28, str(MODELS_DIR / "model_28d.pkl"))
        results["28d"] = {"model": m28, "features": feats_28d, "cv": cv_28d}

        _calibrate_horizon(
            "28d", weekly, feats_28d, "target_28d",
            _fp_ensemble(fn28_1, fn28_2, w28, lam28, hl_weekly),
            "weekly", purge=4, window=None,
        )

    # ── 90-day model (monthly VWAP adj_price, scaled BayesianRidge + recency) ──
    feats_90d = _available_feats(monthly, FEATS_90D, min_non_null=24)
    selected["90d"] = feats_90d
    monthly["target_90d"] = make_return_target(monthly, 3, price_col=pcol_m)
    hl_monthly = RECENCY_HALFLIFE_DAYS["monthly"]

    cv_90d, w90, lam90 = walk_forward_cv_ensemble(
        monthly, feats_90d, "target_90d",
        fn1=_bayesian_90d, fn2=None,
        **WF_CONFIG["monthly"], purge=3,
        anchor_col=pcol_m, halflife_days=hl_monthly,
    )
    _log_cv("90-day", cv_90d)
    blend_map["90d"] = w90
    shrinkage_map["90d"] = lam90

    df_tr = monthly.dropna(subset=["target_90d"])
    if len(df_tr) > 20:
        sw = _recency_weights(df_tr["date"].values, hl_monthly)
        X_tr, y_tr = df_tr[feats_90d].values, df_tr["target_90d"].values
        m90_pipe = _fit_pipe(_bayesian_90d(), X_tr, y_tr, sample_weight=sw)
        m90 = BlendedReturnModel(m90_pipe, None, w1=1.0, lam=lam90)
        joblib.dump(m90, str(MODELS_DIR / "model_90d.pkl"))
        results["90d"] = {"model": m90, "features": feats_90d, "cv": cv_90d}

        _calibrate_horizon(
            "90d", monthly, feats_90d, "target_90d",
            _fp_ensemble(_bayesian_90d, None, 1.0, lam90, hl_monthly),
            "monthly", purge=3, window=None,
        )

    # ── Regime classifier (calibrated L2 LogisticRegression on monthly) ──
    monthly["fwd_6m_ret"] = (monthly[pcol_m].shift(-6) / monthly[pcol_m] - 1.0) * 100.0
    monthly["bear_signal"] = (monthly["fwd_6m_ret"] < -10.0).astype(float)
    monthly.loc[monthly["fwd_6m_ret"].isna(), "bear_signal"] = np.nan

    feats_regime = _available_feats(monthly, FEATS_REGIME, min_non_null=24)
    selected["regime"] = feats_regime

    cv_reg = walk_forward_auc(
        monthly, feats_regime, "bear_signal",
        _logreg_regime, **WF_CONFIG["monthly"], purge=6,
        halflife_days=hl_monthly,
    )
    log.info(f"Regime WF-CV (LogReg): AUC={cv_reg['auc']:.3f}, Brier={cv_reg['brier']:.3f}")

    df_tr = monthly.dropna(subset=["bear_signal"])
    if len(df_tr) > 20 and len(np.unique(df_tr["bear_signal"])) > 1:
        sw = _recency_weights(df_tr["date"].values, hl_monthly)
        m_regime = _fit_pipe(
            _logreg_regime(),
            df_tr[feats_regime].values,
            df_tr["bear_signal"].values,
            sample_weight=sw,
        )
        joblib.dump(m_regime, str(MODELS_DIR / "model_regime.pkl"))
        results["regime"] = {"model": m_regime, "features": feats_regime, "cv": cv_reg}
        log.info(f"Regime model trained on {len(df_tr)} monthly rows")

    joblib.dump(conformal, str(MODELS_DIR / "conformal.pkl"))
    _reseed_aci_state(conformal)
    joblib.dump({
        "model_version": MODEL_VERSION,
        "target": "log_return_adj_v3",
        "features": selected,
        "shrinkage": shrinkage_map,
        "blend_weights": blend_map,
        "has_conformal": bool(conformal),
    }, str(MODELS_DIR / "meta.pkl"))

    results["conformal"] = conformal
    return results


# ── Load saved models ────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all saved v3.0 models from disk (forces retrain on older versions)."""
    ensure_dirs()
    meta_path = MODELS_DIR / "meta.pkl"
    if not meta_path.exists():
        log.warning("No model metadata found — retraining required")
        return {}
    try:
        meta = joblib.load(str(meta_path))
    except Exception as e:
        log.warning(f"Could not read model metadata ({e}) — retraining required")
        return {}
    if meta.get("target") != "log_return_adj_v3":
        log.warning(
            f"Saved models use target {meta.get('target')!r} (expected 'log_return_adj_v3') "
            "— retraining required"
        )
        return {}

    models = {}
    try:
        for name in ["1d", "2d", "3d", "4d", "5d", "6d", "7d", "14d", "28d", "90d", "regime"]:
            path = MODELS_DIR / f"model_{name}.pkl"
            if path.exists():
                models[name] = {"model": joblib.load(str(path))}
    except Exception as e:
        log.warning(f"Failed to load saved models ({e}) — retraining required")
        return {}

    feats_map = meta.get("features", {})
    for key in models:
        group = "short" if key in ("1d", "2d", "3d", "4d", "5d", "6d", "7d") else key
        if group in feats_map:
            models[key]["features"] = feats_map[group]

    conf_path = MODELS_DIR / "conformal.pkl"
    if conf_path.exists():
        try:
            models["conformal"] = joblib.load(str(conf_path))
        except Exception as e:
            log.warning(f"Could not load conformal bands ({e})")

    return models


# ── Online ACI state reader ──────────────────────────────────────────────

def _reseed_aci_state(conformal: dict) -> None:
    """Reset the live Online ACI state to each horizon's freshly calibrated alpha_t.

    Without this, validate.py cold-starts alpha_t at CONFORMAL_ALPHA (0.20)
    rather than the calibrated value, and once a conformal_state row exists it
    overrides every later retrain's calibration for good. Calibration already
    ends in a prequential ACI alpha over the full history, so each retrain
    restarts the online state there and ACI adapts from it between retrains.
    """
    rows = []
    for key, band in (conformal or {}).items():
        if not band or "alpha_t" not in band or not str(key).endswith("d"):
            continue
        try:
            rows.append((int(str(key)[:-1]), round(float(band["alpha_t"]), 5)))
        except ValueError:
            continue
    if not rows:
        return
    try:
        conn = get_conn()
        conn.executemany(
            """INSERT OR REPLACE INTO conformal_state
               (horizon_days, alpha_t, n_updates, empirical_coverage, updated_at)
               VALUES (?, ?, 0, NULL, datetime('now'))""",
            rows,
        )
        conn.commit()
        conn.close()
        log.info(f"Reseeded Online ACI state for {len(rows)} horizons from calibration")
    except Exception as e:  # a state write must never sink a completed retrain
        log.warning(f"Could not reseed Online ACI state ({e})")


def _load_aci_alphas() -> dict[int, float]:
    """Read live per-horizon Online ACI alpha_t overrides from SQLite."""
    try:
        conn = get_conn()
        rows = conn.execute("SELECT horizon_days, alpha_t FROM conformal_state").fetchall()
        conn.close()
        return {int(h): float(a) for h, a in rows if a is not None}
    except Exception:
        return {}


# ── Term-structure coherence ─────────────────────────────────────────────

def _reconcile_term_structure(entries: list[dict], anchor: float) -> list[dict]:
    """Enforce smooth return curve across short horizons and non-decreasing band widths.

    Prevents disconnected single-horizon models from producing jagged reversals
    (e.g. +1.5% at 7d, -2.5% at 14d, +1.5% at 28d) or shrinking uncertainty bands
    at longer horizons.
    """
    if len(entries) < 3 or anchor <= 0:
        return entries

    entries = sorted(entries, key=lambda e: e["horizon_days"])
    rets = [float(np.log(e["predicted_price"] / anchor)) for e in entries]

    # Gentle 3-point weighted smoothing on internal points (0.2, 0.6, 0.2)
    smoothed = list(rets)
    for idx in range(1, len(rets) - 1):
        # Only smooth closely-spaced horizons (ratio <= 2.5x)
        h_prev = entries[idx - 1]["horizon_days"]
        h_cur = entries[idx]["horizon_days"]
        h_next = entries[idx + 1]["horizon_days"]
        if h_next / max(h_cur, 1) <= 2.5 and h_cur / max(h_prev, 1) <= 2.5:
            smoothed[idx] = 0.20 * rets[idx - 1] + 0.60 * rets[idx] + 0.20 * rets[idx + 1]

    prev_lo_dist = 0.0
    prev_hi_dist = 0.0
    for idx, entry in enumerate(entries):
        old_ret = rets[idx]
        new_ret = smoothed[idx]
        shift = new_ret - old_ret
        entry["predicted_price"] = round(anchor * float(np.exp(new_ret)), 1)

        if "lower_bound" in entry and "upper_bound" in entry:
            lo_ret = float(np.log(max(entry["lower_bound"], 1.0) / anchor)) + shift
            hi_ret = float(np.log(max(entry["upper_bound"], 1.0) / anchor)) + shift
            lo_dist = max(new_ret - lo_ret, prev_lo_dist * 0.98)
            hi_dist = max(hi_ret - new_ret, prev_hi_dist * 0.98)
            entry["lower_bound"] = round(anchor * float(np.exp(new_ret - lo_dist)), 1)
            entry["upper_bound"] = round(anchor * float(np.exp(new_ret + hi_dist)), 1)
            prev_lo_dist = lo_dist
            prev_hi_dist = hi_dist

    return entries


# ── Generate predictions ─────────────────────────────────────────────────

def predict_all(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    models: dict,
) -> dict:
    """Generate coherent, volatility-scaled ACI predictions for all horizons."""
    today = daily["date"].max()
    forecasts = {"forecast_date": today, "predictions": []}

    conf_bands = models.get("conformal", {}) or {}
    aci_alphas = _load_aci_alphas()

    pcol_d = _price_col(daily)
    priced_daily = daily.dropna(subset=[pcol_d])
    spot_adj_anchor = float(priced_daily[pcol_d].iloc[-1]) if len(priced_daily) else None

    def _last_row(df: pd.DataFrame, feats: list[str], pcol: str):
        available = [f for f in feats if f in df.columns]
        priced = df.dropna(subset=[pcol])
        if len(priced) == 0 or not available:
            return None, None, None
        row = priced[available].ffill().iloc[-1:].copy()
        # None lets resolve_return_band fall back to the band's calibration
        # median vol; the daily-scale floor would collapse weekly/monthly bands.
        vol = (
            float(priced["rvol_long"].ffill().iloc[-1])
            if "rvol_long" in priced.columns and priced["rvol_long"].notna().any()
            else None
        )
        return row.values, float(priced[pcol].iloc[-1]), vol

    def _attach_band(entry: dict, key: str, h_days: int, ret: float, anchor: float, vol: float):
        band = conf_bands.get(key)
        if not band:
            return
        alpha_ov = aci_alphas.get(h_days)
        q_lo, q_hi = resolve_return_band(band, current_vol=vol, alpha_override=alpha_ov)
        entry["lower_bound"] = round(anchor * float(np.exp(ret + q_lo)), 1)
        entry["upper_bound"] = round(anchor * float(np.exp(ret + q_hi)), 1)
        entry["interval_confidence"] = round(1.0 - band.get("alpha", CONFORMAL_ALPHA), 2)
        entry["interval_method"] = "vol_scaled_aci"

    # 1d–7d & 14d (daily frame)
    for h in list(range(1, 8)) + [14]:
        key = f"{h}d"
        if key not in models:
            continue
        feats = models[key].get("features") or _available_feats(
            daily, FEATS_14D if h == 14 else FEATS_SHORT
        )
        X, anchor, vol = _last_row(daily, feats, pcol_d)
        if X is None:
            continue
        ret = float(models[key]["model"].predict(X)[0])
        pred = anchor * float(np.exp(ret))
        target_date = (pd.Timestamp(today) + timedelta(days=h)).strftime("%Y-%m-%d")
        entry = {
            "horizon_days": h,
            "target_date": target_date,
            "predicted_price": round(pred, 1),
        }
        _attach_band(entry, key, h, ret, anchor, vol)
        forecasts["predictions"].append(entry)
        log.info(f"{h}-day forecast: ₹{pred:.0f} for {target_date}")

    # 28d (weekly VWAP adj_price model, anchored on latest spot adj_price for coherence)
    if "28d" in models:
        pcol_w = _price_col(weekly)
        feats = models["28d"].get("features") or _available_feats(weekly, FEATS_28D)
        X, w_anchor, vol = _last_row(weekly, feats, pcol_w)
        if X is not None:
            anchor = spot_adj_anchor if spot_adj_anchor is not None else w_anchor
            ret = float(models["28d"]["model"].predict(X)[0])
            pred = anchor * float(np.exp(ret))
            target_date = (pd.Timestamp(today) + timedelta(days=28)).strftime("%Y-%m-%d")
            entry = {
                "horizon_days": 28,
                "target_date": target_date,
                "predicted_price": round(pred, 1),
            }
            _attach_band(entry, "28d", 28, ret, anchor, vol)
            forecasts["predictions"].append(entry)
            log.info(f"28-day forecast: ₹{pred:.0f} for {target_date}")

    # 90d (monthly VWAP adj_price model, anchored on latest spot adj_price)
    if "90d" in models:
        pcol_m = _price_col(monthly)
        feats = models["90d"].get("features") or _available_feats(monthly, FEATS_90D)
        X, m_anchor, vol = _last_row(monthly, feats, pcol_m)
        if X is not None:
            anchor = spot_adj_anchor if spot_adj_anchor is not None else m_anchor
            ret = float(models["90d"]["model"].predict(X)[0])
            pred = anchor * float(np.exp(ret))
            target_date = (pd.Timestamp(today) + timedelta(days=90)).strftime("%Y-%m-%d")
            entry = {
                "horizon_days": 90,
                "target_date": target_date,
                "predicted_price": round(pred, 1),
            }
            _attach_band(entry, "90d", 90, ret, anchor, vol)
            forecasts["predictions"].append(entry)
            log.info(
                f"90-day forecast: ₹{pred:.0f} "
                f"[{entry.get('lower_bound', 0):.0f}–{entry.get('upper_bound', 0):.0f}] "
                f"for {target_date}"
            )

    # Term-structure smoothing & monotonic band widths
    if spot_adj_anchor and forecasts["predictions"]:
        forecasts["predictions"] = _reconcile_term_structure(
            forecasts["predictions"], spot_adj_anchor
        )

    # Regime (calibrated 6m bear probability)
    if "regime" in models:
        pcol_m = _price_col(monthly)
        feats = models["regime"].get("features") or _available_feats(monthly, FEATS_REGIME)
        X, _, _ = _last_row(monthly, feats, pcol_m)
        if X is not None:
            prob = float(models["regime"]["model"].predict_proba(X)[0, 1])
            label = "LOW" if prob < 0.30 else ("MODERATE" if prob < 0.60 else "HIGH")
            forecasts["regime"] = {
                "bear_probability": round(prob, 3),
                "label": label,
            }
            log.info(f"Regime: {prob:.1%} bear probability ({label})")

    _sanity_check_forecasts(forecasts, daily)
    return forecasts


def _sanity_check_forecasts(forecasts: dict, daily: pd.DataFrame) -> None:
    """Abort the run if the shortest-horizon forecast deviates implausibly from spot."""
    priced = daily["avg_price"].dropna()
    preds = forecasts.get("predictions", [])
    if len(priced) == 0 or not preds:
        return
    spot = float(priced.iloc[-1])
    nearest = min(preds, key=lambda p: p["horizon_days"])
    dev = abs(nearest["predicted_price"] / spot - 1.0)
    if dev > MAX_SHORT_HORIZON_DEVIATION:
        raise ValueError(
            f"{nearest['horizon_days']}-day forecast "
            f"₹{nearest['predicted_price']:.0f} deviates {dev:.1%} from spot "
            f"₹{spot:.0f} (limit {MAX_SHORT_HORIZON_DEVIATION:.0%}) — aborting before publish."
        )
