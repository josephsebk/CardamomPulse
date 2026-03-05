#!/usr/bin/env python3
"""
Backtest: Compare baseline (T1+T2 only) vs enhanced (T1+T2+T3+T4) feature sets
for all short-horizon models (1d–14d).

Also backtests 90d and regime models with T4 added.
"""

import logging
import numpy as np
import pandas as pd
from pipeline.assemble import build_daily_df, resample_weekly, resample_monthly
from pipeline.features import (
    add_tier1, add_tier2, add_tier3, add_tier4,
    T1_FEATURES, T2_FEATURES_DAILY, T3_FEATURES, T4_FEATURES,
    T2_FEATURES_MONTHLY, T5_FEATURES, T6_FEATURES,
)
from pipeline.models import walk_forward_cv, _gbr_short, _gbr_7d, _gbr_14d
from pipeline.models import _bayesian_90d, _gbc_regime, FEATS_90D, FEATS_REGIME
from pipeline.config import WF_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Build data ────────────────────────────────────────────────────────────
log.info("Assembling data...")
daily_raw = build_daily_df()
monthly_raw = resample_monthly(daily_raw)

# ── Prepare daily DataFrame with ALL tiers ────────────────────────────────
daily = daily_raw.copy()
add_tier1(daily)
add_tier2(daily)
add_tier3(daily, is_monthly=False)
add_tier4(daily)

# ── Prepare monthly DataFrame with ALL tiers ──────────────────────────────
monthly = monthly_raw.copy()
add_tier1(monthly, vol_col="sold_kg", arr_col="arrived_kg")
add_tier2(monthly)
add_tier3(monthly, is_monthly=True)
add_tier4(monthly)
# T5 and T6 for 90d/regime
from pipeline.features import add_tier5, add_tier6
add_tier5(monthly)
add_tier6(monthly)


def available(df, feat_list):
    return [f for f in feat_list if f in df.columns]


# ── Define feature sets ───────────────────────────────────────────────────

# BASELINE: what the models currently use
baseline_feats_short = available(daily, T1_FEATURES + T2_FEATURES_DAILY[:5])  # 1d-7d
baseline_feats_14d   = available(daily, T1_FEATURES + T2_FEATURES_DAILY)      # 14d

# ENHANCED: add weather (T3) + macro (T4)
enhanced_feats_short = available(daily, T1_FEATURES + T2_FEATURES_DAILY[:5] + T3_FEATURES + T4_FEATURES)
enhanced_feats_14d   = available(daily, T1_FEATURES + T2_FEATURES_DAILY + T3_FEATURES + T4_FEATURES)

# WEATHER ONLY variant: T1 + T2 + T3
weather_feats_short  = available(daily, T1_FEATURES + T2_FEATURES_DAILY[:5] + T3_FEATURES)
weather_feats_14d    = available(daily, T1_FEATURES + T2_FEATURES_DAILY + T3_FEATURES)

# MACRO ONLY variant: T1 + T2 + T4
macro_feats_short    = available(daily, T1_FEATURES + T2_FEATURES_DAILY[:5] + T4_FEATURES)
macro_feats_14d      = available(daily, T1_FEATURES + T2_FEATURES_DAILY + T4_FEATURES)

# ── 90d and Regime: baseline vs enhanced with T4 ─────────────────────────
baseline_90d   = available(monthly, FEATS_90D)  # current: no T4
enhanced_90d   = available(monthly, FEATS_90D + T4_FEATURES)  # add T4

baseline_regime = available(monthly, FEATS_REGIME)  # current: no T4
enhanced_regime = available(monthly, FEATS_REGIME + T4_FEATURES)  # add T4


# ── Run walk-forward CV ──────────────────────────────────────────────────
wf_daily = WF_CONFIG["daily"]
wf_monthly = WF_CONFIG["monthly"]

print("\n" + "=" * 90)
print("BACKTEST: Baseline vs Enhanced Feature Sets — Walk-Forward Cross-Validation")
print("=" * 90)

# ── Short horizons (1d–7d) ───────────────────────────────────────────────
print(f"\n{'Horizon':<10} {'Variant':<20} {'#Feats':>6} {'MAPE':>8} {'MAE':>10} {'R²':>8} {'Folds':>6}")
print("-" * 70)

results = []

for h in [1, 2, 3, 5, 7]:
    target_col = f"target_{h}d"
    daily[target_col] = daily["avg_price"].shift(-h)

    model_fn = (lambda _h=h: _gbr_short(_h)) if h <= 6 else _gbr_7d

    for variant, feats in [
        ("Baseline (T1+T2)", baseline_feats_short),
        ("+Weather (T3)", weather_feats_short),
        ("+Macro (T4)", macro_feats_short),
        ("+Both (T3+T4)", enhanced_feats_short),
    ]:
        cv = walk_forward_cv(daily, feats, target_col, model_fn,
                             **wf_daily, purge=h)
        label = f"{h}d"
        print(f"{label:<10} {variant:<20} {len(feats):>6} {cv['mape']:>8.3f} {cv['mae']:>10.1f} {cv['r2']:>8.3f} {cv['folds']:>6}")
        results.append({"horizon": label, "variant": variant,
                        "n_feats": len(feats), **cv})
    print()

# ── 14-day ────────────────────────────────────────────────────────────────
daily["target_14d"] = daily["avg_price"].shift(-14)
for variant, feats in [
    ("Baseline (T1+T2)", baseline_feats_14d),
    ("+Weather (T3)", weather_feats_14d),
    ("+Macro (T4)", macro_feats_14d),
    ("+Both (T3+T4)", enhanced_feats_14d),
]:
    cv = walk_forward_cv(daily, feats, "target_14d", _gbr_14d,
                         **wf_daily, purge=14)
    print(f"{'14d':<10} {variant:<20} {len(feats):>6} {cv['mape']:>8.3f} {cv['mae']:>10.1f} {cv['r2']:>8.3f} {cv['folds']:>6}")
    results.append({"horizon": "14d", "variant": variant,
                    "n_feats": len(feats), **cv})

# ── 90-day ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("90-day model (BayesianRidge):")
print("-" * 70)
monthly["target_90d"] = monthly["avg_price"].shift(-3)
medians_base = monthly[baseline_90d].median()
monthly_base = monthly.copy()
monthly_base[baseline_90d] = monthly_base[baseline_90d].fillna(medians_base)

medians_enh = monthly[enhanced_90d].median()
monthly_enh = monthly.copy()
monthly_enh[enhanced_90d] = monthly_enh[enhanced_90d].fillna(medians_enh)

from sklearn.preprocessing import StandardScaler

for variant, feats, mdf in [
    ("Baseline (no T4)", baseline_90d, monthly_base),
    ("+Macro (T4)", enhanced_90d, monthly_enh),
]:
    # Scale features for BayesianRidge
    def make_bayesian():
        return _bayesian_90d()

    cv = walk_forward_cv(mdf, feats, "target_90d", make_bayesian,
                         **wf_monthly, purge=3)
    print(f"{'90d':<10} {variant:<20} {len(feats):>6} {cv['mape']:>8.3f} {cv['mae']:>10.1f} {cv['r2']:>8.3f} {cv['folds']:>6}")
    results.append({"horizon": "90d", "variant": variant,
                    "n_feats": len(feats), **cv})

# ── Regime classifier ────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("Regime classifier (GradientBoostingClassifier):")
print("-" * 70)
monthly["fwd_6m_ret"] = (monthly["avg_price"].shift(-6) / monthly["avg_price"] - 1) * 100
monthly["bear_signal"] = (monthly["fwd_6m_ret"] < -10).astype(int)

for variant, feats in [
    ("Baseline (no T4)", baseline_regime),
    ("+Macro (T4)", enhanced_regime),
]:
    med = monthly[feats].median()
    mdf = monthly.copy()
    mdf[feats] = mdf[feats].fillna(med)

    # Classification accuracy via walk-forward
    df_clean = mdf.dropna(subset=["bear_signal"] + feats)
    X = df_clean[feats].values
    y = df_clean["bear_signal"].values
    n = len(df_clean)

    all_preds, all_actuals = [], []
    i = wf_monthly["min_train"]
    while i + wf_monthly["eval_win"] <= n:
        train_end = i - 3
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[i:i + wf_monthly["eval_win"]], y[i:i + wf_monthly["eval_win"]]
        if len(X_tr) < 20 or len(X_te) == 0:
            i += wf_monthly["step"]
            continue
        m = _gbc_regime()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        all_preds.extend(preds)
        all_actuals.extend(y_te)
        i += wf_monthly["step"]

    if all_preds:
        acc = np.mean(np.array(all_preds) == np.array(all_actuals))
        print(f"{'regime':<10} {variant:<20} {len(feats):>6} {'accuracy':>8} {acc:>10.3f}")
    results.append({"horizon": "regime", "variant": variant,
                    "n_feats": len(feats), "accuracy": acc if all_preds else None})


# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("IMPROVEMENT SUMMARY (MAPE reduction = better)")
print("=" * 90)
print(f"\n{'Horizon':<10} {'Baseline MAPE':>14} {'Enhanced MAPE':>14} {'Improvement':>12} {'Better?':>8}")
print("-" * 60)

for h in ["1d", "2d", "3d", "5d", "7d", "14d", "90d"]:
    base = [r for r in results if r["horizon"] == h and "Baseline" in r["variant"]]
    enh = [r for r in results if r["horizon"] == h and "Both" in r["variant"] or
           (r["horizon"] == h and "+Macro" in r["variant"] and h == "90d")]
    if base and enh:
        b_mape = base[0]["mape"]
        e_mape = enh[-1]["mape"]
        improvement = (b_mape - e_mape) / b_mape * 100
        better = "YES" if e_mape < b_mape else "no"
        print(f"{h:<10} {b_mape:>14.3f} {e_mape:>14.3f} {improvement:>11.1f}% {better:>8}")

print("\nDone.")
