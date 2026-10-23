"""Split-conformal prediction intervals for the forecast models.

The regression models predict log-returns and price is reconstructed as
``anchor * exp(return)``. We wrap those point forecasters with conformal
intervals calibrated on **walk-forward out-of-sample residuals** measured on
the return scale, then map the band through the same ``exp()`` so the price
interval scales multiplicatively with the current price level — the right
behaviour for a series with a ~9x peak-to-trough range.

Two deliberate choices:

* **Asymmetric bands.** The lower and upper residual quantiles are taken
  separately rather than a symmetric ``±|resid|``. Cardamom returns have fat
  upside tails (supply shocks spike prices far more than they fall), so a
  symmetric band is miscalibrated on both sides at once.

* **Recency window (adaptive).** Bands are calibrated on the most recent
  ``window`` residuals rather than all history. During a volatile regime the
  recent residuals grow and the band widens automatically; in calm periods it
  tightens. This is the cheap, batch-pipeline-friendly form of adaptive
  conformal — it reacts to the local error distribution without needing the
  streaming coverage feedback loop of full online ACI.

Calibrating on pooled walk-forward residuals (which already span multiple
price regimes) is the closest we get to valid marginal coverage on a
non-stationary series; ``prequential_coverage`` reports the empirically
achieved coverage so the calibration can be checked rather than assumed.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Minimum calibration residuals below which a band is not trustworthy.
MIN_CAL_RESID = 20


def walk_forward_residuals(df, features, target_col, fit_predict,
                           min_train, step, eval_win, purge=0):
    """Collect signed out-of-sample residuals on the target (log-return) scale.

    Mirrors the walk-forward loop in ``models.walk_forward_cv`` but returns the
    pooled ``actual - predicted`` residuals in chronological order instead of
    aggregate metrics. ``fit_predict(X_train, y_train, X_test)`` must train on
    the fold's training window and return predicted returns for ``X_test`` —
    this lets each horizon supply the *exact* predictor it deploys (single GBR,
    the 28d GBR+Ridge stack, or the 90d Bayesian ridge), so the residual
    distribution matches the model the band will wrap.
    """
    d = df.dropna(subset=[target_col] + features)
    X = d[features].values
    y = d[target_col].values
    n = len(d)

    residuals = []
    i = min_train
    while i + eval_win <= n:
        train_end = i - purge
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[i: i + eval_win], y[i: i + eval_win]
        if len(X_train) < 20 or len(X_test) == 0:
            i += step
            continue
        try:
            preds = np.asarray(fit_predict(X_train, y_train, X_test), dtype=float)
        except Exception as e:  # a bad fold should not sink calibration
            log.warning(f"residual fold at i={i} failed ({e}); skipping")
            i += step
            continue
        residuals.extend(np.asarray(y_test, dtype=float) - preds)
        i += step

    return np.asarray(residuals, dtype=float)


def conformal_band(residuals, alpha=0.2, window=None):
    """Asymmetric split-conformal band on the residual (log-return) scale.

    Returns ``(q_lo, q_hi)`` such that a point forecast ``r̂`` yields the
    return-scale interval ``[r̂ + q_lo, r̂ + q_hi]`` with target coverage
    ``1 - alpha``. ``q_lo`` is typically negative and ``q_hi`` positive.

    Uses order-statistic quantiles with the standard finite-sample inflation
    (rank ``ceil((1 - alpha/2)(n+1))`` for the upper tail), so coverage is
    distribution-free and slightly conservative on small calibration sets.
    Returns ``None`` when there are too few residuals to calibrate.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if window is not None and len(r) > window:
        r = r[-window:]
    n = len(r)
    if n < MIN_CAL_RESID:
        return None

    sorted_r = np.sort(r)
    # Upper tail: smallest residual whose rank covers (1 - alpha/2).
    k_hi = int(np.ceil((1 - alpha / 2) * (n + 1)))
    q_hi = sorted_r[min(k_hi, n) - 1]
    # Lower tail: residual at the alpha/2 quantile.
    k_lo = int(np.floor((alpha / 2) * (n + 1)))
    q_lo = sorted_r[max(k_lo, 1) - 1]
    return float(q_lo), float(q_hi)


def prequential_coverage(residuals, alpha=0.2, window=None, min_cal=100):
    """Validate the band by replaying the residual sequence in order.

    At each step ``t`` the band is calibrated on the residuals seen *before*
    ``t`` (optionally only the most recent ``window``) and we record whether
    the realized residual at ``t`` fell inside it. This is an out-of-sample,
    no-look-ahead estimate of the coverage the deployed band actually delivers
    on this series — the honest counterpart to the nominal ``1 - alpha``.

    Returns a dict with empirical ``coverage``, ``mean_width_pct`` (mean band
    width expressed as a percentage of price, via ``exp``), and the count of
    evaluated points.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    start = max(min_cal, MIN_CAL_RESID)
    if n <= start:
        return {"coverage": float("nan"), "mean_width_pct": float("nan"), "n": 0}

    hits = 0
    widths = []
    count = 0
    for t in range(start, n):
        cal = r[:t]
        band = conformal_band(cal, alpha=alpha, window=window)
        if band is None:
            continue
        q_lo, q_hi = band
        e = r[t]
        if q_lo <= e <= q_hi:
            hits += 1
        # Band width as a fraction of price: (exp(q_hi) - exp(q_lo)).
        widths.append(np.exp(q_hi) - np.exp(q_lo))
        count += 1

    if count == 0:
        return {"coverage": float("nan"), "mean_width_pct": float("nan"), "n": 0}
    return {
        "coverage": hits / count,
        "mean_width_pct": float(np.mean(widths) * 100),
        "n": count,
    }


def calibrate(residuals, alpha=0.2, window=None, min_cal=100):
    """Build the persisted conformal record for one horizon.

    Combines the deployed band (``q_lo``/``q_hi`` from the recency window) with
    the prequential validation result, so a single dict carries both what
    ``predict_all`` applies and the coverage it was shown to achieve.
    """
    band = conformal_band(residuals, alpha=alpha, window=window)
    if band is None:
        return None
    q_lo, q_hi = band
    val = prequential_coverage(residuals, alpha=alpha, window=window,
                               min_cal=min_cal)
    return {
        "q_lo": q_lo,
        "q_hi": q_hi,
        "alpha": float(alpha),
        "window": window,
        "coverage": val["coverage"],
        "mean_width_pct": val["mean_width_pct"],
        "n_resid": int(np.isfinite(residuals).sum()),
    }
