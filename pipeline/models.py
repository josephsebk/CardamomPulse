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

from pipeline.config import MODELS_DIR, MODEL_VERSION, WF_CONFIG, ensure_dirs
from pipeline.features import (
    T1_FEATURES, T2_FEATURES_DAILY, T2_FEATURES_MONTHLY,
    T3_FEATURES, T4_FEATURES, T5_FEATURES, T6_FEATURES,
)

log = logging.getLogger(__name__)

# ── Feature sets per horizon ─────────────────────────────────────────────
# We select the subset that each model actually uses (matching the notebook)

# 7-day: T1 + first 5 of T2 daily
FEATS_7D = T1_FEATURES + T2_FEATURES_DAILY[:5]

# 14-day: T1 + all T2 daily
FEATS_14D = T1_FEATURES + T2_FEATURES_DAILY

# 28-day (weekly): T1 + T2 + T3 + T4 + T5 (selected down during training)
FEATS_28D_CANDIDATES = T1_FEATURES + T2_FEATURES_DAILY + T3_FEATURES + T4_FEATURES + T5_FEATURES

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


def _available_feats(df: pd.DataFrame, feat_list: list[str]) -> list[str]:
    """Return only features that exist in the DataFrame."""
    return [f for f in feat_list if f in df.columns]


# ── Walk-Forward Cross-Validation ────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, features: list[str], target_col: str,
                    model_fn, min_train: int, step: int, eval_win: int,
                    purge: int = 0) -> dict:
    """Run walk-forward cross-validation. Returns metrics dict."""
    df_clean = df.dropna(subset=[target_col] + features)
    X = df_clean[features].values
    y = df_clean[target_col].values
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

    # ── 1d–6d daily models (same features as 7d) ──────────────────
    feats_short = _available_feats(daily, FEATS_7D)
    for h in range(1, 7):
        key = f"{h}d"
        target_col = f"target_{key}"
        daily[target_col] = daily["avg_price"].shift(-h)
        cv = walk_forward_cv(daily, feats_short, target_col,
                             lambda _h=h: _gbr_short(_h),
                             **WF_CONFIG["daily"], purge=h)
        log.info(f"{h}-day WF-CV: MAPE={cv['mape']:.3f}, MAE={cv['mae']:.1f}")

        df_train = daily.dropna(subset=[target_col] + feats_short)
        m = _gbr_short(h)
        m.fit(df_train[feats_short].values, df_train[target_col].values)
        joblib.dump(m, str(MODELS_DIR / f"model_{key}.pkl"))
        results[key] = {"model": m, "features": feats_short, "cv": cv}

    # ── 7-day model ──────────────────────────────────────────────────
    feats_7d = _available_feats(daily, FEATS_7D)
    daily["target_7d"] = daily["avg_price"].shift(-7)
    cv_7d = walk_forward_cv(daily, feats_7d, "target_7d", _gbr_7d,
                            **WF_CONFIG["daily"], purge=7)
    log.info(f"7-day WF-CV: MAPE={cv_7d['mape']:.3f}, MAE={cv_7d['mae']:.1f}")

    df_train = daily.dropna(subset=["target_7d"] + feats_7d)
    m7 = _gbr_7d()
    m7.fit(df_train[feats_7d].values, df_train["target_7d"].values)
    joblib.dump(m7, str(MODELS_DIR / "model_7d.pkl"))
    results["7d"] = {"model": m7, "features": feats_7d, "cv": cv_7d}

    # ── 14-day model ─────────────────────────────────────────────────
    feats_14d = _available_feats(daily, FEATS_14D)
    daily["target_14d"] = daily["avg_price"].shift(-14)
    cv_14d = walk_forward_cv(daily, feats_14d, "target_14d", _gbr_14d,
                             **WF_CONFIG["daily"], purge=14)
    log.info(f"14-day WF-CV: MAPE={cv_14d['mape']:.3f}, MAE={cv_14d['mae']:.1f}")

    df_train = daily.dropna(subset=["target_14d"] + feats_14d)
    m14 = _gbr_14d()
    m14.fit(df_train[feats_14d].values, df_train["target_14d"].values)
    joblib.dump(m14, str(MODELS_DIR / "model_14d.pkl"))
    results["14d"] = {"model": m14, "features": feats_14d, "cv": cv_14d}

    # ── 28-day model (stacked GBM + Ridge) ───────────────────────────
    feats_28d = _available_feats(weekly, FEATS_28D_CANDIDATES)
    weekly["target_28d"] = weekly["avg_price"].shift(-4)  # 4 weeks forward
    cv_28d = walk_forward_cv(weekly, feats_28d, "target_28d", _gbr_28d,
                             **WF_CONFIG["weekly"], purge=4)
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
    feats_90d = _available_feats(monthly, FEATS_90D)
    monthly["target_90d"] = monthly["avg_price"].shift(-3)

    # Impute NaN with column medians for monthly models (small dataset)
    medians_90 = monthly[feats_90d].median()
    monthly_90 = monthly.copy()
    monthly_90[feats_90d] = monthly_90[feats_90d].fillna(medians_90)

    cv_90d = walk_forward_cv(monthly_90, feats_90d, "target_90d",
                             _bayesian_90d, **WF_CONFIG["monthly"], purge=3)
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
    feats_regime = _available_feats(monthly, FEATS_REGIME)
    monthly["fwd_6m_ret"] = (monthly["avg_price"].shift(-6) / monthly["avg_price"] - 1) * 100
    monthly["bear_signal"] = (monthly["fwd_6m_ret"] < -10).astype(int)

    # Impute NaN with column medians
    medians_reg = monthly[feats_regime].median()
    monthly_reg = monthly.copy()
    monthly_reg[feats_regime] = monthly_reg[feats_regime].fillna(medians_reg)

    df_train = monthly_reg.dropna(subset=["bear_signal"])
    if len(df_train) > 20:
        X_reg = df_train[feats_regime].values
        y_reg = df_train["bear_signal"].values
        m_regime = _gbc_regime()
        m_regime.fit(X_reg, y_reg)
        joblib.dump(m_regime, str(MODELS_DIR / "model_regime.pkl"))
        joblib.dump(medians_reg.to_dict(), str(MODELS_DIR / "medians_regime.pkl"))
        results["regime"] = {"model": m_regime, "features": feats_regime,
                             "medians": medians_reg.to_dict()}
        log.info(f"Regime model trained on {len(df_train)} rows")

    return results


# ── Load saved models ────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all saved models from disk."""
    ensure_dirs()
    models = {}

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

    return models


# ── Generate predictions ─────────────────────────────────────────────────

def predict_all(daily: pd.DataFrame, weekly: pd.DataFrame,
                monthly: pd.DataFrame, models: dict) -> dict:
    """Generate predictions from the latest data row for each model."""
    today = daily["date"].max()
    forecasts = {"forecast_date": today, "predictions": []}

    # Helper to get last non-NaN feature row
    def _last_row(df, feats):
        available = [f for f in feats if f in df.columns]
        sub = df[available].dropna()
        if len(sub) == 0:
            return None, available
        return sub.iloc[-1:].values, available

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

    # 1d–6d daily forecasts
    for h in range(1, 7):
        key = f"{h}d"
        if key in models:
            feats = _available_feats(daily, FEATS_7D)
            X, used = _last_row(daily, feats)
            if X is not None:
                pred = float(models[key]["model"].predict(X)[0])
                target_date = (pd.Timestamp(today) + timedelta(days=h)).strftime("%Y-%m-%d")
                forecasts["predictions"].append({
                    "horizon_days": h, "target_date": target_date,
                    "predicted_price": round(pred, 1),
                })
                log.info(f"{h}-day forecast: ₹{pred:.0f} for {target_date}")

    # 7-day
    if "7d" in models:
        feats = _available_feats(daily, FEATS_7D)
        X, used = _last_row(daily, feats)
        if X is not None:
            pred = float(models["7d"]["model"].predict(X)[0])
            target_date = (pd.Timestamp(today) + timedelta(days=7)).strftime("%Y-%m-%d")
            forecasts["predictions"].append({
                "horizon_days": 7, "target_date": target_date,
                "predicted_price": round(pred, 1),
            })
            log.info(f"7-day forecast: ₹{pred:.0f} for {target_date}")

    # 14-day
    if "14d" in models:
        feats = _available_feats(daily, FEATS_14D)
        X, used = _last_row(daily, feats)
        if X is not None:
            pred = float(models["14d"]["model"].predict(X)[0])
            target_date = (pd.Timestamp(today) + timedelta(days=14)).strftime("%Y-%m-%d")
            forecasts["predictions"].append({
                "horizon_days": 14, "target_date": target_date,
                "predicted_price": round(pred, 1),
            })
            log.info(f"14-day forecast: ₹{pred:.0f} for {target_date}")

    # 28-day (stacked)
    if "28d" in models:
        feats = _available_feats(weekly, FEATS_28D_CANDIDATES)
        X, used = _last_row(weekly, feats)
        if X is not None:
            p_lgb = float(models["28d"]["model_lgb"].predict(X)[0])
            X_sc = models["28d"]["scaler"].transform(X)
            p_ridge = float(models["28d"]["model_ridge"].predict(X_sc)[0])
            pred = 0.5 * p_lgb + 0.5 * p_ridge
            target_date = (pd.Timestamp(today) + timedelta(days=28)).strftime("%Y-%m-%d")
            forecasts["predictions"].append({
                "horizon_days": 28, "target_date": target_date,
                "predicted_price": round(pred, 1),
            })
            log.info(f"28-day forecast: ₹{pred:.0f} for {target_date}")

    # 90-day (with prediction intervals)
    if "90d" in models:
        feats = _available_feats(monthly, FEATS_90D)
        X = _last_row_imputed(monthly, feats, models["90d"].get("medians", {}))
        if X is not None:
            X_sc = models["90d"]["scaler"].transform(X)
            p_mean, p_std = models["90d"]["model"].predict(X_sc, return_std=True)
            pred = float(p_mean[0])
            lo = float(pred - 1.28 * p_std[0])  # 80% lower
            hi = float(pred + 1.28 * p_std[0])  # 80% upper
            target_date = (pd.Timestamp(today) + timedelta(days=90)).strftime("%Y-%m-%d")
            forecasts["predictions"].append({
                "horizon_days": 90, "target_date": target_date,
                "predicted_price": round(pred, 1),
                "lower_bound": round(lo, 1),
                "upper_bound": round(hi, 1),
            })
            log.info(f"90-day forecast: ₹{pred:.0f} [{lo:.0f}–{hi:.0f}] for {target_date}")

    # Regime
    if "regime" in models:
        feats = _available_feats(monthly, FEATS_REGIME)
        X = _last_row_imputed(monthly, feats, models["regime"].get("medians", {}))
        if X is not None:
            prob = float(models["regime"]["model"].predict_proba(X)[0, 1])
            label = "LOW" if prob < 0.3 else ("MODERATE" if prob < 0.6 else "HIGH")
            forecasts["regime"] = {
                "bear_probability": round(prob, 3),
                "label": label,
            }
            log.info(f"Regime: {prob:.1%} bear probability ({label})")

    return forecasts
