"""Volatility-scaled Split-Conformal & Online Adaptive Conformal Inference (ACI).

v3.0 upgrades over v2.4:
1. **Locally-weighted (volatility-scaled) conformal scores.**
   Signed return residuals are normalized by trailing realized volatility
   `sigma_t = max(rvol_t, CONFORMAL_VOL_FLOOR)`:
       s_t = (y_t - y_hat_t) / sigma_t
   Asymmetric quantiles (q_lo, q_hi) of `s_t` are multiplied back by current
   realized volatility `sigma_now` at inference:
       [r_hat + sigma_now * q_lo, r_hat + sigma_now * q_hi]
   So intervals tighten automatically in calm regimes and widen immediately when
   auction volatility spikes.

2. **Online Adaptive Conformal Inference (Gibbs & Candès, 2021).**
   Each horizon maintains a persisted effective miscoverage rate `alpha_t` in
   SQLite (`conformal_state`). Whenever `validate.py` settles a matured forecast,
   `update_aci_alpha` nudges `alpha_t`:
       alpha_{t+1} = clip(alpha_t + gamma * (alpha_target - miss_t), min, max)
   Because the recent calibration score array is stored in `conformal.pkl`,
   online ACI adapts interval widths *daily* without waiting for weekly retrains.
"""

import logging
import numpy as np

from pipeline.config import (
    CONFORMAL_ACI_GAMMA,
    CONFORMAL_ALPHA,
    CONFORMAL_ALPHA_MAX,
    CONFORMAL_ALPHA_MIN,
    CONFORMAL_VOL_FLOOR,
)

log = logging.getLogger(__name__)

MIN_CAL_RESID = 20


def _sanitize_vols(vols, n: int, floor: float = CONFORMAL_VOL_FLOOR) -> np.ndarray:
    """Return strictly positive finite volatility scale factors."""
    if vols is None:
        return np.ones(n, dtype=float)
    v = np.asarray(vols, dtype=float)
    if len(v) != n:
        return np.ones(n, dtype=float)
    med = float(np.nanmedian(v[np.isfinite(v) & (v > 0)])) if np.any(np.isfinite(v) & (v > 0)) else floor
    v = np.where(np.isfinite(v) & (v > 0), v, med)
    return np.clip(v, floor, None)


def walk_forward_residuals(
    df,
    features,
    target_col,
    fit_predict,
    min_train,
    step,
    eval_win,
    purge=0,
    vol_col: str | None = "rvol_long",
    date_col: str | None = "date",
):
    """Collect chronological out-of-sample (residual, volatility) arrays.

    `fit_predict(X_train, y_train, X_test, dates_train=None)` replicates the
    exact deployed predictor (including imputation, scaling, recency weights,
    and shrinkage). Returns `(residuals, vols)`.
    """
    d = df.dropna(subset=[target_col]).copy()
    if len(d) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    X = d[features].values
    y = d[target_col].values
    vols_all = (
        d[vol_col].values if (vol_col and vol_col in d.columns) else np.ones(len(d), dtype=float)
    )
    dates_all = (
        d[date_col].values if (date_col and date_col in d.columns) else None
    )
    n = len(d)

    residuals, vols_out = [], []
    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[i: i + eval_win], y[i: i + eval_win]
        if len(X_train) < 20 or len(X_test) == 0:
            i += step
            continue
        try:
            if dates_all is not None:
                preds = np.asarray(
                    fit_predict(X_train, y_train, X_test, dates_all[:train_end]),
                    dtype=float,
                )
            else:
                preds = np.asarray(fit_predict(X_train, y_train, X_test), dtype=float)
        except TypeError:
            preds = np.asarray(fit_predict(X_train, y_train, X_test), dtype=float)
        except Exception as e:
            log.warning(f"residual fold at i={i} failed ({e}); skipping")
            i += step
            continue

        residuals.extend(np.asarray(y_test, dtype=float) - preds)
        vols_out.extend(np.asarray(vols_all[i: i + eval_win], dtype=float))
        i += step

    return np.asarray(residuals, dtype=float), np.asarray(vols_out, dtype=float)


def conformal_band(scores, alpha=CONFORMAL_ALPHA, window=None):
    """Asymmetric split-conformal quantiles `(q_lo, q_hi)` on scores (or residuals)."""
    r = np.asarray(scores, dtype=float)
    r = r[np.isfinite(r)]
    if window is not None and len(r) > window:
        r = r[-window:]
    n = len(r)
    if n < MIN_CAL_RESID:
        return None

    a = float(np.clip(alpha, CONFORMAL_ALPHA_MIN, CONFORMAL_ALPHA_MAX))
    sorted_r = np.sort(r)
    k_hi = int(np.ceil((1.0 - a / 2.0) * (n + 1)))
    q_hi = sorted_r[min(max(k_hi, 1), n) - 1]
    k_lo = int(np.floor((a / 2.0) * (n + 1)))
    q_lo = sorted_r[min(max(k_lo, 1), n) - 1]
    return float(q_lo), float(q_hi)


def update_aci_alpha(
    alpha_t: float,
    in_interval: bool,
    target_alpha: float = CONFORMAL_ALPHA,
    gamma: float = CONFORMAL_ACI_GAMMA,
) -> float:
    """One online Adaptive Conformal Inference (ACI) step.

    When an actual price misses the interval (`miss = 1`), `alpha_t` decreases
    so future bands widen; when it hits (`miss = 0`), `alpha_t` gently increases
    so bands tighten back toward nominal efficiency.
    """
    miss = 0.0 if in_interval else 1.0
    next_alpha = float(alpha_t) + float(gamma) * (float(target_alpha) - miss)
    return float(np.clip(next_alpha, CONFORMAL_ALPHA_MIN, CONFORMAL_ALPHA_MAX))


def prequential_coverage(
    residuals,
    vols=None,
    alpha=CONFORMAL_ALPHA,
    window=None,
    min_cal=100,
    gamma=CONFORMAL_ACI_GAMMA,
):
    """Validate volatility-scaled + online ACI bands prequentially (strictly causal)."""
    r = np.asarray(residuals, dtype=float)
    v = _sanitize_vols(vols, len(r))
    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    scores = r / v

    n = len(r)
    start = max(min_cal, MIN_CAL_RESID)
    if n <= start:
        return {"coverage": float("nan"), "mean_width_pct": float("nan"), "n": 0, "final_alpha": float(alpha)}

    hits = 0
    widths = []
    count = 0
    alpha_t = float(alpha)

    for t in range(start, n):
        band = conformal_band(scores[:t], alpha=alpha_t, window=window)
        if band is None:
            continue
        q_lo_s, q_hi_s = band
        lo = q_lo_s * v[t]
        hi = q_hi_s * v[t]
        in_band = (lo <= r[t] <= hi)
        if in_band:
            hits += 1
        widths.append(np.exp(hi) - np.exp(lo))
        count += 1
        if gamma and gamma > 0:
            alpha_t = update_aci_alpha(alpha_t, in_band, target_alpha=alpha, gamma=gamma)

    if count == 0:
        return {"coverage": float("nan"), "mean_width_pct": float("nan"), "n": 0, "final_alpha": float(alpha)}
    return {
        "coverage": hits / count,
        "mean_width_pct": float(np.mean(widths) * 100.0),
        "n": count,
        "final_alpha": float(alpha_t),
    }


def calibrate(
    residuals,
    vols=None,
    alpha=CONFORMAL_ALPHA,
    window=None,
    min_cal=100,
    gamma=CONFORMAL_ACI_GAMMA,
):
    """Build the persisted volatility-scaled conformal record for one horizon.

    Stores both the default normalized quantiles `(q_lo_scaled, q_hi_scaled)` and
    the bounded calibration score array `scores` so online ACI can recompute
    quantiles dynamically whenever `alpha_t` updates from live validation_log.
    """
    r = np.asarray(residuals, dtype=float)
    v = _sanitize_vols(vols, len(r))
    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    if len(r) < MIN_CAL_RESID:
        return None

    scores = r / v
    if window is not None and len(scores) > window:
        cal_scores = scores[-window:]
        cal_vols = v[-window:]
    else:
        cal_scores = scores
        cal_vols = v

    val = prequential_coverage(r, v, alpha=alpha, window=window, min_cal=min_cal, gamma=gamma)
    active_alpha = val["final_alpha"] if np.isfinite(val.get("final_alpha", np.nan)) else float(alpha)

    band_s = conformal_band(cal_scores, alpha=active_alpha, window=None)
    if band_s is None:
        return None
    q_lo_s, q_hi_s = band_s
    med_vol = float(np.median(cal_vols))

    return {
        "scaled": True,
        "q_lo_scaled": q_lo_s,
        "q_hi_scaled": q_hi_s,
        # Fallback unscaled quantiles evaluated at median volatility (backward compat)
        "q_lo": float(q_lo_s * med_vol),
        "q_hi": float(q_hi_s * med_vol),
        "median_vol": med_vol,
        "vol_floor": float(CONFORMAL_VOL_FLOOR),
        "scores": [round(float(x), 5) for x in cal_scores],
        "alpha": float(alpha),
        "alpha_t": float(active_alpha),
        "window": window,
        "coverage": val["coverage"],
        "mean_width_pct": val["mean_width_pct"],
        "n_resid": int(len(r)),
    }


def resolve_return_band(
    band: dict,
    current_vol: float | None = None,
    alpha_override: float | None = None,
) -> tuple[float, float]:
    """Compute `(q_lo_ret, q_hi_ret)` on the log-return scale for inference."""
    if not band:
        return 0.0, 0.0

    # Legacy unscaled v2.4 record fallback
    if not band.get("scaled"):
        return float(band["q_lo"]), float(band["q_hi"])

    vol_floor = float(band.get("vol_floor", CONFORMAL_VOL_FLOOR))
    med_vol = float(band.get("median_vol", vol_floor))
    vol = (
        float(current_vol)
        if (current_vol is not None and np.isfinite(current_vol) and current_vol > 0)
        else med_vol
    )
    vol = max(vol, vol_floor)

    scores = band.get("scores")
    eff_alpha = alpha_override if alpha_override is not None else band.get("alpha_t", band.get("alpha", CONFORMAL_ALPHA))
    if scores and len(scores) >= MIN_CAL_RESID:
        qs = conformal_band(scores, alpha=eff_alpha, window=None)
        if qs is not None:
            q_lo_s, q_hi_s = qs
            return float(q_lo_s * vol), float(q_hi_s * vol)

    return float(band["q_lo_scaled"] * vol), float(band["q_hi_scaled"] * vol)
