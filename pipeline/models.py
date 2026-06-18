"""Model training, walk-forward CV, and prediction."""

import logging
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    MODELS_DIR, MODEL_VERSION, WF_CONFIG, FEATURE_SELECTION_K,
    MAX_SHORT_HORIZON_DEVIATION, MAX_ANCHOR_AGE_DAYS, ensure_dirs,
)
from pipeline.features import (
    T1_FEATURES, MICRO_FEATURES, T2_FEATURES_DAILY, T2_FEATURES_MONTHLY,
    T3_FEATURES, T4_FEATURES, T5_FEATURES, T6_FEATURES,
)

log = logging.getLogger(__name__)

# ── Feature sets per horizon ─────────────────────────────────────────────
# Hand-curated fallback sets, used when FEATURE_SELECTION_K disables
# selection for a horizon or when no selected list was saved with a model.

# 1d–7d: T1 + first 5 of T2 daily + T4 macro
FEATS_7D = T1_FEATURES + T2_FEATURES_DAILY[:5] + T4_FEATURES

# 14-day: T1 + all T2 daily + T4 macro
FEATS_14D = T1_FEATURES + T2_FEATURES_DAILY + T4_FEATURES

# 28-day (weekly): T1 + microstructure + T2 + T3 + T4 + T5
FEATS_28D_CANDIDATES = (
    T1_FEATURES + MICRO_FEATURES + T2_FEATURES_DAILY
    + T3_FEATURES + T4_FEATURES + T5_FEATURES
)

# 90-day (monthly): T1 subset + T2 monthly + T3 tail + T5 + T6
FEATS_90D = (
    T1_FEATURES[:10]
    + T2_FEATURES_MONTHLY[:2]
    + T3_FEATURES[-5:]
    + T5_FEATURES
    + T6_FEATURES
)

# Regime: T1 subset + T3 + T5 + T6
FEATS_REGIME = T1_FEATURES[:10] + T3_FEATURES + T5_FEATURES + T6_FEATURES

# Candidate pools for walk-forward permutation feature selection.
# Microstructure is daily/weekly only: monthly averaging destroys these
# short-lived signals, and including them degraded regime AUC in backtests.
FEATS_DAILY_CANDIDATES = (
    T1_FEATURES + MICRO_FEATURES + T2_FEATURES_DAILY + T3_FEATURES + T4_FEATURES
)
FEATS_MONTHLY_CANDIDATES = (
    T1_FEATURES + T2_FEATURES_MONTHLY + T3_FEATURES + T5_FEATURES + T6_FEATURES
)


def _available_feats(df: pd.DataFrame, feat_list: list[str]) -> list[str]:
    """Return only features that exist in the DataFrame."""
    return [f for f in feat_list if f in df.columns]


# ── Targets ──────────────────────────────────────────────────────────────

def make_return_target(df: pd.DataFrame, horizon: int,
                       price_col: str = "avg_price") -> pd.Series:
    """Log-return target: log(price[t+h] / price[t]).

    Returns are roughly stationary across price regimes, so tree models can
    forecast levels outside the training range (predicted price is
    reconstructed as price[t] * exp(predicted return)).
    """
    return np.log(df[price_col].shift(-horizon) / df[price_col])


# ── Walk-Forward Cross-Validation ────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, features: list[str], target_col: str,
                    model_fn, min_train: int, step: int, eval_win: int,
                    purge: int = 0, anchor_col: str | None = None) -> dict:
    """Run walk-forward cross-validation. Returns metrics dict.

    If anchor_col is given, the target is interpreted as a log-return
    relative to anchor_col and metrics are computed on reconstructed price
    levels, keeping MAPE/MAE comparable with level-target runs.
    """
    subset = [target_col] + features + ([anchor_col] if anchor_col else [])
    df_clean = df.dropna(subset=subset)
    X = df_clean[features].values
    y = df_clean[target_col].values
    anchors = df_clean[anchor_col].values if anchor_col else None
    n = len(df_clean)

    all_preds, all_actuals = [], []
    fold = 0

    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[i: i + eval_win], y[i: i + eval_win]

        if len(X_train) < 20 or len(X_test) == 0:
            i += step
            continue

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if anchors is not None:
            a = anchors[i: i + eval_win]
            preds = a * np.exp(preds)
            y_test = a * np.exp(y_test)

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        fold += 1
        i += step

    if not all_preds:
        return {"mape": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "folds": 0}

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    errors = np.abs(preds - actuals)
    pct_errors = errors / np.abs(actuals).clip(min=1)

    mape = float(np.mean(pct_errors))
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-10))

    return {"mape": mape, "mae": mae, "rmse": rmse, "r2": r2, "folds": fold}


def walk_forward_auc(df: pd.DataFrame, features: list[str], target_col: str,
                     model_fn, min_train: int, step: int, eval_win: int,
                     purge: int = 0) -> dict:
    """Walk-forward CV for binary classifiers. Returns AUC/Brier on pooled
    out-of-sample probabilities."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    df_clean = df.dropna(subset=[target_col] + features)
    X = df_clean[features].values
    y = df_clean[target_col].values
    n = len(df_clean)

    probs, actuals = [], []
    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        if train_end >= 20 and len(np.unique(y[:train_end])) > 1:
            model = model_fn()
            model.fit(X[:train_end], y[:train_end])
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


# ── Walk-forward permutation feature selection ──────────────────────────

def select_features(df: pd.DataFrame, candidates: list[str], target_col: str,
                    model_fn, min_train: int, step: int, eval_win: int,
                    purge: int = 0, top_k: int = 20,
                    n_repeats: int = 5) -> list[str]:
    """Rank candidate features by mean permutation importance on the held-out
    window of each walk-forward fold; return the top_k.

    Importance measured out-of-sample avoids the trap of impurity-based
    rankings, which favor high-cardinality features the model overfits on.
    """
    from sklearn.inspection import permutation_importance

    df_clean = df.dropna(subset=[target_col] + candidates)
    X = df_clean[candidates].values
    y = df_clean[target_col].values
    n = len(df_clean)

    importances = np.zeros(len(candidates))
    folds = 0
    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[i: i + eval_win], y[i: i + eval_win]

        if len(X_train) < 20 or len(X_test) == 0 or (
                len(np.unique(y_train)) < 2):
            i += step
            continue

        model = model_fn()
        model.fit(X_train, y_train)
        r = permutation_importance(model, X_test, y_test,
                                   n_repeats=n_repeats, random_state=42)
        importances += r.importances_mean
        folds += 1
        i += step

    if folds == 0:
        return candidates[:top_k]

    order = np.argsort(importances)[::-1]
    return [candidates[j] for j in order[:top_k]]


# ── Model factories ──────────────────────────────────────────────────────

def _gbr_short(horizon):
    """Factory for short-horizon (1d–7d) GBR models."""
    # Shorter horizons use slightly less complexity
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=4,
        learning_rate=0.12 if horizon <= 3 else 0.1,
        subsample=0.8,
        min_samples_leaf=max(5, horizon * 2),
        random_state=42,
    )


def _gbr_7d():
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )


def _gbr_14d():
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=15, random_state=42,
    )


def _gbr_28d():
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )


def _ridge_28d():
    return Ridge(alpha=10.0)


def _bayesian_90d():
    return BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
    )


def _gbc_regime():
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


# ── Train all models ─────────────────────────────────────────────────────

def train_all(daily: pd.DataFrame, weekly: pd.DataFrame,
              monthly: pd.DataFrame) -> dict:
    """Train all models on full data and save to disk. Returns metrics + models."""
    ensure_dirs()
    results = {}
    selected = {}  # horizon-group -> feature list, persisted in meta.pkl

    # ── Short-horizon feature set (shared by 1d–7d, ranked on 7d) ────
    daily["target_7d"] = make_return_target(daily, 7)
    k_short = FEATURE_SELECTION_K.get("short")
    if k_short:
        cand = _available_feats(daily, FEATS_DAILY_CANDIDATES)
        feats_short = select_features(daily, cand, "target_7d", _gbr_7d,
                                      **WF_CONFIG["daily"], purge=7,
                                      top_k=k_short)
        log.info(f"short-horizon selected top-{k_short}: {feats_short}")
    else:
        feats_short = _available_feats(daily, FEATS_7D)
    selected["short"] = feats_short

    # ── 1d–6d daily models (same features as 7d) ──────────────────
    for h in range(1, 7):
        key = f"{h}d"
        target_col = f"target_{key}"
        daily[target_col] = make_return_target(daily, h)
        cv = walk_forward_cv(daily, feats_short, target_col,
                             lambda _h=h: _gbr_short(_h),
                             **WF_CONFIG["daily"], purge=h,
                             anchor_col="avg_price")
        log.info(f"{h}-day WF-CV: MAPE={cv['mape']:.3f}, MAE={cv['mae']:.1f}")

        df_train = daily.dropna(subset=[target_col] + feats_short)
        m = _gbr_short(h)
        m.fit(df_train[feats_short].values, df_train[target_col].values)
        joblib.dump(m, str(MODELS_DIR / f"model_{key}.pkl"))
        results[key] = {"model": m, "features": feats_short, "cv": cv}

    # ── 7-day model ──────────────────────────────────────────────────
    feats_7d = feats_short
    cv_7d = walk_forward_cv(daily, feats_7d, "target_7d", _gbr_7d,
                            **WF_CONFIG["daily"], purge=7,
                            anchor_col="avg_price")
    log.info(f"7-day WF-CV: MAPE={cv_7d['mape']:.3f}, MAE={cv_7d['mae']:.1f}")

    df_train = daily.dropna(subset=["target_7d"] + feats_7d)
    m7 = _gbr_7d()
    m7.fit(df_train[feats_7d].values, df_train["target_7d"].values)
    joblib.dump(m7, str(MODELS_DIR / "model_7d.pkl"))
    results["7d"] = {"model": m7, "features": feats_7d, "cv": cv_7d}

    # ── 14-day model ─────────────────────────────────────────────────
    daily["target_14d"] = make_return_target(daily, 14)
    k_14d = FEATURE_SELECTION_K.get("14d")
    if k_14d:
        cand = _available_feats(daily, FEATS_DAILY_CANDIDATES)
        feats_14d = select_features(daily, cand, "target_14d", _gbr_14d,
                                    **WF_CONFIG["daily"], purge=14,
                                    top_k=k_14d)
        log.info(f"14d selected top-{k_14d}: {feats_14d}")
    else:
        feats_14d = _available_feats(daily, FEATS_14D)
    selected["14d"] = feats_14d
    cv_14d = walk_forward_cv(daily, feats_14d, "target_14d", _gbr_14d,
                             **WF_CONFIG["daily"], purge=14,
                             anchor_col="avg_price")
    log.info(f"14-day WF-CV: MAPE={cv_14d['mape']:.3f}, MAE={cv_14d['mae']:.1f}")

    df_train = daily.dropna(subset=["target_14d"] + feats_14d)
    m14 = _gbr_14d()
    m14.fit(df_train[feats_14d].values, df_train["target_14d"].values)
    joblib.dump(m14, str(MODELS_DIR / "model_14d.pkl"))
    results["14d"] = {"model": m14, "features": feats_14d, "cv": cv_14d}

    # ── 28-day model (stacked GBM + Ridge) ───────────────────────────
    # No selection here: the manual T1-T5 set beat every selected subset
    feats_28d = _available_feats(weekly, FEATS_28D_CANDIDATES)
    selected["28d"] = feats_28d
    weekly["target_28d"] = make_return_target(weekly, 4)  # 4 weeks forward
    cv_28d = walk_forward_cv(weekly, feats_28d, "target_28d", _gbr_28d,
                             **WF_CONFIG["weekly"], purge=4,
                             anchor_col="avg_price")
    log.info(f"28-day WF-CV: MAPE={cv_28d['mape']:.3f}, MAE={cv_28d['mae']:.1f}")

    df_train = weekly.dropna(subset=["target_28d"] + feats_28d)
    if len(df_train) > 20:
        X_28 = df_train[feats_28d].values
        y_28 = df_train["target_28d"].values

        m28_lgb = _gbr_28d()
        m28_lgb.fit(X_28, y_28)

        sc28 = StandardScaler()
        X_28_sc = sc28.fit_transform(X_28)
        m28_ridge = _ridge_28d()
        m28_ridge.fit(X_28_sc, y_28)

        joblib.dump(m28_lgb, str(MODELS_DIR / "model_28d_lgb.pkl"))
        joblib.dump(m28_ridge, str(MODELS_DIR / "model_28d_ridge.pkl"))
        joblib.dump(sc28, str(MODELS_DIR / "scaler_28d.pkl"))
        results["28d"] = {
            "model_lgb": m28_lgb, "model_ridge": m28_ridge,
            "scaler": sc28, "features": feats_28d, "cv": cv_28d,
        }

    # ── 90-day model (Bayesian Ridge) ────────────────────────────────
    monthly["target_90d"] = make_return_target(monthly, 3)

    # Impute NaN with column medians for monthly models (small dataset)
    cand_monthly = _available_feats(monthly, FEATS_MONTHLY_CANDIDATES)
    medians_all = monthly[cand_monthly].median()
    monthly_90 = monthly.copy()
    monthly_90[cand_monthly] = monthly_90[cand_monthly].fillna(medians_all)

    k_90d = FEATURE_SELECTION_K.get("90d")
    if k_90d:
        feats_90d = select_features(monthly_90, cand_monthly, "target_90d",
                                    _bayesian_90d, **WF_CONFIG["monthly"],
                                    purge=3, top_k=k_90d)
        log.info(f"90d selected top-{k_90d}: {feats_90d}")
    else:
        feats_90d = _available_feats(monthly, FEATS_90D)
    selected["90d"] = feats_90d
    medians_90 = medians_all[feats_90d]

    cv_90d = walk_forward_cv(monthly_90, feats_90d, "target_90d",
                             _bayesian_90d, **WF_CONFIG["monthly"], purge=3,
                             anchor_col="avg_price")
    log.info(f"90-day WF-CV: MAPE={cv_90d['mape']:.3f}, MAE={cv_90d['mae']:.1f}")

    df_train = monthly_90.dropna(subset=["target_90d"])
    if len(df_train) > 20:
        sc90 = StandardScaler()
        X_90 = sc90.fit_transform(df_train[feats_90d].values)
        m90 = _bayesian_90d()
        m90.fit(X_90, df_train["target_90d"].values)
        joblib.dump(m90, str(MODELS_DIR / "model_90d.pkl"))
        joblib.dump(sc90, str(MODELS_DIR / "scaler_90d.pkl"))
        joblib.dump(medians_90.to_dict(), str(MODELS_DIR / "medians_90d.pkl"))
        results["90d"] = {"model": m90, "scaler": sc90, "features": feats_90d,
                          "medians": medians_90.to_dict(), "cv": cv_90d}

    # ── Regime classifier ────────────────────────────────────────────
    monthly["fwd_6m_ret"] = (monthly["avg_price"].shift(-6) / monthly["avg_price"] - 1) * 100
    # Rows whose 6-month outcome is still unknown must be excluded from
    # training, not labelled 0 (NaN < -10 evaluates to False)
    monthly["bear_signal"] = (monthly["fwd_6m_ret"] < -10).astype(float)
    monthly.loc[monthly["fwd_6m_ret"].isna(), "bear_signal"] = np.nan

    # Impute NaN with column medians
    monthly_reg = monthly.copy()
    monthly_reg[cand_monthly] = monthly_reg[cand_monthly].fillna(medians_all)

    k_reg = FEATURE_SELECTION_K.get("regime")
    if k_reg:
        feats_regime = select_features(monthly_reg, cand_monthly,
                                       "bear_signal", _gbc_regime,
                                       **WF_CONFIG["monthly"], purge=6,
                                       top_k=k_reg)
        log.info(f"regime selected top-{k_reg}: {feats_regime}")
    else:
        feats_regime = _available_feats(monthly, FEATS_REGIME)
    selected["regime"] = feats_regime
    medians_reg = medians_all[feats_regime]

    cv_reg = walk_forward_auc(monthly_reg, feats_regime, "bear_signal",
                              _gbc_regime, **WF_CONFIG["monthly"], purge=6)
    log.info(f"Regime WF-CV: AUC={cv_reg['auc']:.3f}, Brier={cv_reg['brier']:.3f}")

    df_train = monthly_reg.dropna(subset=["bear_signal"])
    if len(df_train) > 20:
        X_reg = df_train[feats_regime].values
        y_reg = df_train["bear_signal"].values
        m_regime = _gbc_regime()
        m_regime.fit(X_reg, y_reg)
        joblib.dump(m_regime, str(MODELS_DIR / "model_regime.pkl"))
        joblib.dump(medians_reg.to_dict(), str(MODELS_DIR / "medians_regime.pkl"))
        results["regime"] = {"model": m_regime, "features": feats_regime,
                             "medians": medians_reg.to_dict(), "cv": cv_reg}
        log.info(f"Regime model trained on {len(df_train)} rows")

    # Marker so load_models() can reject pickles trained on a different
    # target definition (pre-v2.0 models predicted price levels), plus the
    # per-horizon feature lists the saved models were trained with
    joblib.dump({"model_version": MODEL_VERSION, "target": "log_return",
                 "features": selected},
                str(MODELS_DIR / "meta.pkl"))

    return results


# ── Load saved models ────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all saved models from disk."""
    ensure_dirs()

    # Regression models predict log-returns since v2.0; refuse to load
    # older pickles whose outputs are price levels — exp() of a level
    # would produce garbage forecasts. Returning {} triggers a retrain.
    meta_path = MODELS_DIR / "meta.pkl"
    if not meta_path.exists():
        log.warning("No model metadata found — saved models predate the "
                    "log-return target; retraining required")
        return {}
    try:
        meta = joblib.load(str(meta_path))
    except Exception as e:
        log.warning(f"Could not read model metadata ({e}) — retraining required")
        return {}
    if meta.get("target") != "log_return":
        log.warning(f"Saved models use target {meta.get('target')!r} — "
                    "retraining required")
        return {}

    models = {}

    # Pickles can become unloadable when a dependency upgrade changes
    # internal module paths (e.g. sklearn upgrade in CI broke cached models
    # on 2026-06-02..06). Treat any load failure as "no models" so the
    # caller retrains instead of crashing the pipeline.
    try:
        for name, files in [
            ("1d", ["model_1d.pkl"]),
            ("2d", ["model_2d.pkl"]),
            ("3d", ["model_3d.pkl"]),
            ("4d", ["model_4d.pkl"]),
            ("5d", ["model_5d.pkl"]),
            ("6d", ["model_6d.pkl"]),
            ("7d", ["model_7d.pkl"]),
            ("14d", ["model_14d.pkl"]),
            ("28d", ["model_28d_lgb.pkl", "model_28d_ridge.pkl", "scaler_28d.pkl"]),
            ("90d", ["model_90d.pkl", "scaler_90d.pkl"]),
            ("regime", ["model_regime.pkl"]),
        ]:
            all_exist = all((MODELS_DIR / f).exists() for f in files)
            if not all_exist:
                continue

            if name in ("1d", "2d", "3d", "4d", "5d", "6d"):
                models[name] = {"model": joblib.load(str(MODELS_DIR / f"model_{name}.pkl"))}
            elif name == "7d":
                models["7d"] = {"model": joblib.load(str(MODELS_DIR / "model_7d.pkl"))}
            elif name == "14d":
                models["14d"] = {"model": joblib.load(str(MODELS_DIR / "model_14d.pkl"))}
            elif name == "28d":
                models["28d"] = {
                    "model_lgb": joblib.load(str(MODELS_DIR / "model_28d_lgb.pkl")),
                    "model_ridge": joblib.load(str(MODELS_DIR / "model_28d_ridge.pkl")),
                    "scaler": joblib.load(str(MODELS_DIR / "scaler_28d.pkl")),
                }
            elif name == "90d":
                m = {
                    "model": joblib.load(str(MODELS_DIR / "model_90d.pkl")),
                    "scaler": joblib.load(str(MODELS_DIR / "scaler_90d.pkl")),
                }
                med_path = MODELS_DIR / "medians_90d.pkl"
                if med_path.exists():
                    m["medians"] = joblib.load(str(med_path))
                models["90d"] = m
            elif name == "regime":
                m = {"model": joblib.load(str(MODELS_DIR / "model_regime.pkl"))}
                med_path = MODELS_DIR / "medians_regime.pkl"
                if med_path.exists():
                    m["medians"] = joblib.load(str(med_path))
                models["regime"] = m
    except Exception as e:
        log.warning(f"Failed to load saved models ({e}) — retraining required")
        return {}

    # Attach the feature lists the saved models were trained with
    feats_map = meta.get("features", {})
    for key in models:
        group = "short" if key in ("1d", "2d", "3d", "4d", "5d", "6d", "7d") else key
        if group in feats_map:
            models[key]["features"] = feats_map[group]

    return models


# ── Generate predictions ─────────────────────────────────────────────────

def predict_all(daily: pd.DataFrame, weekly: pd.DataFrame,
                monthly: pd.DataFrame, models: dict) -> dict:
    """Generate predictions from the latest data row for each model."""
    today = daily["date"].max()
    forecasts = {"forecast_date": today, "predictions": []}

    # ── Data freshness gate ────────────────────────────────────────────────
    # If the latest priced row is older than MAX_ANCHOR_AGE_DAYS relative to
    # the system date, the scraper has been failing and every prediction will
    # be silently anchored to a stale price level. Short-horizon models
    # (≤7d) are useless in this state; we skip them and add a staleness flag
    # so the webapp can warn users rather than show misleading numbers.
    import datetime as _dt
    _anchor_date = pd.Timestamp(today).date()
    _sys_date = _dt.date.today()
    _anchor_age = (_sys_date - _anchor_date).days
    _data_stale = _anchor_age > MAX_ANCHOR_AGE_DAYS
    if _data_stale:
        log.warning(
            f"Anchor date {_anchor_date} is {_anchor_age} days old "
            f"(limit {MAX_ANCHOR_AGE_DAYS}). Short-horizon forecasts "
            f"(≤7d) suppressed; longer horizons flagged as stale."
        )
        forecasts["data_stale"] = True
        forecasts["anchor_date"] = str(_anchor_date)
        forecasts["anchor_age_days"] = _anchor_age
    else:
        forecasts["data_stale"] = False

    # Helper: features and anchor price for the latest priced row.
    # The anchor MUST be the most recent actual price — a log-return forecast
    # is reconstructed as anchor * exp(return), so anchoring on an older row
    # silently shifts every forecast toward that row's (stale) price level.
    # Earlier this conflated "latest price" with "row where every selected
    # feature is non-NaN": a NaN in a selected feature (e.g. microstructure
    # missing on recently-scraped days) rewound the anchor months back and
    # produced phantom multi-percent jumps. We now always take the last
    # priced row and forward-fill / median-impute any residual NaN features.
    def _last_row(df, feats):
        available = [f for f in feats if f in df.columns]
        priced = df.dropna(subset=["avg_price"])
        if len(priced) == 0:
            return None, available, None
        # causal carry-forward so the latest row inherits the last known
        # value of any feature that is missing on it
        row = priced[available].ffill().iloc[-1:].copy()
        stale = [c for c in available if row[c].isna().iloc[0]]
        if stale:
            log.warning(f"Features still NaN on latest row after ffill, "
                        f"imputing 0: {stale}")
            row[stale] = 0.0
        return (row.values, available, float(priced["avg_price"].iloc[-1]))

    # Helper: get last row with median imputation for NaN
    def _last_row_imputed(df, feats, medians):
        available = [f for f in feats if f in df.columns]
        if not available:
            return None
        row = df[available].iloc[-1:].copy()
        for col in available:
            if pd.isna(row[col].iloc[0]) and col in medians:
                row[col] = medians[col]
        return row.values

    # 1d–6d daily forecasts — suppressed when data is stale
    if _data_stale:
        log.warning("Skipping 1d–6d forecasts: anchor data too old.")
    else:
        for h in range(1, 7):
            key = f"{h}d"
            if key in models:
                feats = models[key].get("features") or _available_feats(daily, FEATS_7D)
                X, used, anchor = _last_row(daily, feats)
                if X is not None:
                    ret = float(models[key]["model"].predict(X)[0])
                    pred = anchor * float(np.exp(ret))
                    target_date = (pd.Timestamp(today) + timedelta(days=h)).strftime("%Y-%m-%d")
                    forecasts["predictions"].append({
                        "horizon_days": h, "target_date": target_date,
                        "predicted_price": round(pred, 1),
                    })
                log.info(f"{h}-day forecast: ₹{pred:.0f} for {target_date}")

    # 7-day — also suppressed when data is stale
    if not _data_stale and "7d" in models:
        feats = models["7d"].get("features") or _available_feats(daily, FEATS_7D)
        X, used, anchor = _last_row(daily, feats)
        if X is not None:
            ret = float(models["7d"]["model"].predict(X)[0])
            pred = anchor * float(np.exp(ret))
            target_date = (pd.Timestamp(today) + timedelta(days=7)).strftime("%Y-%m-%d")
            forecasts["predictions"].append({
                "horizon_days": 7, "target_date": target_date,
                "predicted_price": round(pred, 1),
            })
            log.info(f"7-day forecast: ₹{pred:.0f} for {target_date}")

    # 14-day
    if "14d" in models:
        feats = models["14d"].get("features") or _available_feats(daily, FEATS_14D)
        X, used, anchor = _last_row(daily, feats)
        if X is not None:
            ret = float(models["14d"]["model"].predict(X)[0])
            pred = anchor * float(np.exp(ret))
            target_date = (pd.Timestamp(today) + timedelta(days=14)).strftime("%Y-%m-%d")
            entry = {"horizon_days": 14, "target_date": target_date,
                     "predicted_price": round(pred, 1)}
            if _data_stale:
                entry["stale_data"] = True
            forecasts["predictions"].append(entry)
            log.info(f"14-day forecast: ₹{pred:.0f} for {target_date}")

    # 28-day (stacked)
    if "28d" in models:
        feats = models["28d"].get("features") or _available_feats(weekly, FEATS_28D_CANDIDATES)
        X, used, anchor = _last_row(weekly, feats)
        if X is not None:
            p_lgb = float(models["28d"]["model_lgb"].predict(X)[0])
            X_sc = models["28d"]["scaler"].transform(X)
            p_ridge = float(models["28d"]["model_ridge"].predict(X_sc)[0])
            ret = 0.5 * p_lgb + 0.5 * p_ridge
            pred = anchor * float(np.exp(ret))
            target_date = (pd.Timestamp(today) + timedelta(days=28)).strftime("%Y-%m-%d")
            entry = {"horizon_days": 28, "target_date": target_date,
                     "predicted_price": round(pred, 1)}
            if _data_stale:
                entry["stale_data"] = True
            forecasts["predictions"].append(entry)
            log.info(f"28-day forecast: ₹{pred:.0f} for {target_date}")

    # 90-day (with prediction intervals)
    if "90d" in models:
        feats = models["90d"].get("features") or _available_feats(monthly, FEATS_90D)
        X = _last_row_imputed(monthly, feats, models["90d"].get("medians", {}))
        anchor_series = monthly["avg_price"].dropna()
        if X is not None and len(anchor_series) > 0:
            anchor = float(anchor_series.iloc[-1])
            X_sc = models["90d"]["scaler"].transform(X)
            p_mean, p_std = models["90d"]["model"].predict(X_sc, return_std=True)
            ret = float(p_mean[0])
            pred = anchor * float(np.exp(ret))
            lo = anchor * float(np.exp(ret - 1.28 * p_std[0]))  # 80% lower
            hi = anchor * float(np.exp(ret + 1.28 * p_std[0]))  # 80% upper
            target_date = (pd.Timestamp(today) + timedelta(days=90)).strftime("%Y-%m-%d")
            entry = {
                "horizon_days": 90, "target_date": target_date,
                "predicted_price": round(pred, 1),
                "lower_bound": round(lo, 1),
                "upper_bound": round(hi, 1),
            }
            if _data_stale:
                entry["stale_data"] = True
            forecasts["predictions"].append(entry)
            log.info(f"90-day forecast: ₹{pred:.0f} [{lo:.0f}–{hi:.0f}] for {target_date}")

    # Regime
    if "regime" in models:
        feats = models["regime"].get("features") or _available_feats(monthly, FEATS_REGIME)
        X = _last_row_imputed(monthly, feats, models["regime"].get("medians", {}))
        if X is not None:
            prob = float(models["regime"]["model"].predict_proba(X)[0, 1])
            label = "LOW" if prob < 0.3 else ("MODERATE" if prob < 0.6 else "HIGH")
            forecasts["regime"] = {
                "bear_probability": round(prob, 3),
                "label": label,
            }
            log.info(f"Regime: {prob:.1%} bear probability ({label})")

    _sanity_check_forecasts(forecasts, daily)
    return forecasts


def _sanity_check_forecasts(forecasts: dict, daily: pd.DataFrame) -> None:
    """Abort the run if the shortest-horizon forecast deviates implausibly
    from spot. Catches anchor/feature bugs that would otherwise publish a
    phantom move (the stale-anchor bug produced a ~15% 1-day 'crash').

    Note: when data is stale the freshness gate in predict_all() has already
    suppressed ≤7d forecasts, so this check only sees 14d+ entries. It still
    validates them against the (stale) anchor; that's intentional — a 14d
    forecast that deviates >10% from even a stale anchor is still a sign of
    a model bug."""
    priced = daily["avg_price"].dropna()
    preds = [p for p in forecasts.get("predictions", []) if not p.get("stale_data")]
    if len(priced) == 0 or not preds:
        return
    spot = float(priced.iloc[-1])
    # Only check short-horizon forecasts (≤14d) against the deviation limit;
    # longer horizons legitimately move more.
    short_preds = [p for p in preds if p["horizon_days"] <= 14]
    if not short_preds:
        return
    nearest = min(short_preds, key=lambda p: p["horizon_days"])
    dev = abs(nearest["predicted_price"] / spot - 1)
    if dev > MAX_SHORT_HORIZON_DEVIATION:
        raise ValueError(
            f"{nearest['horizon_days']}-day forecast "
            f"₹{nearest['predicted_price']:.0f} deviates {dev:.1%} from spot "
            f"₹{spot:.0f} (limit {MAX_SHORT_HORIZON_DEVIATION:.0%}) — likely a "
            f"stale anchor or feature bug; aborting before publish.")
