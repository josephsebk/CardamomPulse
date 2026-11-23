#!/usr/bin/env python3
"""
Backtest: Compare v3.0 feature tiers (T1+T2 vs +Weather vs +Macro vs full v3.0
horizon-matched feature sets) across 1d–90d and the 6m bear-regime classifier.
"""

import logging

import numpy as np

from pipeline.assemble import build_daily_df, resample_monthly, resample_weekly
from pipeline.config import RECENCY_HALFLIFE_DAYS, WF_CONFIG
from pipeline.features import (
    T1_FEATURES,
    T2_FEATURES_DAILY,
    T2_FEATURES_MONTHLY,
    T3_FEATURES,
    T4_FEATURES,
    T5_FEATURES,
    T6_FEATURES,
    add_tier1,
    add_tier2,
    add_tier3,
    add_tier4,
    add_tier5,
    add_tier6,
)
from pipeline.models import (
    FEATS_14D,
    FEATS_28D,
    FEATS_90D,
    FEATS_REGIME,
    FEATS_SHORT,
    _bayesian_90d,
    _gbr_shallow,
    _huber_short,
    _logreg_regime,
    _ridge_med,
    _ridge_short,
    make_return_target,
    make_return_target_calendar,
    walk_forward_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def available(df, feat_list, min_non_null=20):
    return [f for f in feat_list if f in df.columns and int(df[f].notna().sum()) >= min_non_null]


def main():
    log.info("Assembling data...")
    daily = build_daily_df()
    weekly = resample_weekly(daily)
    monthly = resample_monthly(daily)

    for frame, freq in [(daily, "daily"), (weekly, "weekly"), (monthly, "monthly")]:
        add_tier1(frame, freq=freq)
        add_tier2(frame, freq=freq)
        add_tier3(frame, freq=freq)
        add_tier4(frame, freq=freq)
    add_tier5(monthly)
    add_tier6(monthly, freq="monthly")

    wf_daily = WF_CONFIG["daily"]
    wf_weekly = WF_CONFIG["weekly"]
    wf_monthly = WF_CONFIG["monthly"]

    print("\n" + "=" * 96)
    print("BACKTEST v3.0: Stationary Horizon-Matched Feature Sets — Leak-Free Walk-Forward CV")
    print("=" * 96)
    print(f"\n{'Horizon':<8} {'Variant':<26} {'#Feats':>6} {'MAPE':>8} {'Naive':>8} {'Skill':>8} {'TheilU':>7} {'Folds':>6}")
    print("-" * 82)

    base_short = available(daily, FEATS_SHORT)
    plus_t3 = available(daily, FEATS_SHORT + T3_FEATURES)
    plus_t4 = available(daily, FEATS_SHORT + T4_FEATURES)
    v3_14d = available(daily, FEATS_14D)

    for h in [1, 3, 7]:
        tcol = f"target_{h}d"
        daily[tcol] = make_return_target_calendar(daily, h)
        hl = RECENCY_HALFLIFE_DAYS["daily"]
        for label, feats, fn in [
            ("v3 Short (Huber)", base_short, lambda _h=h: _huber_short(_h)),
            ("v3 Short (Ridge)", base_short, lambda _h=h: _ridge_short(_h)),
            ("+Weather (Ridge)", plus_t3, lambda _h=h: _ridge_short(_h)),
            ("+Macro (Ridge)", plus_t4, lambda _h=h: _ridge_short(_h)),
        ]:
            cv = walk_forward_cv(daily, feats, tcol, fn, **wf_daily, purge=h, halflife_days=hl)
            print(f"{f'{h}d':<8} {label:<26} {len(feats):>6} {cv['mape']:>8.4f} {cv['naive_mape']:>8.4f} "
                  f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} {cv['folds']:>6}")
        print()

    daily["target_14d"] = make_return_target_calendar(daily, 14)
    for label, feats, fn in [
        ("v3 14d (Ridge)", v3_14d, lambda: _ridge_med(alpha=40.0)),
        ("v3 14d (Shallow GBR)", v3_14d, lambda: _gbr_shallow(min_leaf=30, n_estimators=60)),
    ]:
        cv = walk_forward_cv(daily, feats, "target_14d", fn, **wf_daily,
                             purge=14, halflife_days=RECENCY_HALFLIFE_DAYS["daily"])
        print(f"{'14d':<8} {label:<26} {len(feats):>6} {cv['mape']:>8.4f} {cv['naive_mape']:>8.4f} "
              f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} {cv['folds']:>6}")

    weekly["target_28d"] = make_return_target(weekly, 4)
    v3_28d = available(weekly, FEATS_28D)
    print()
    for label, feats, fn in [
        ("v3 28d (Ridge)", v3_28d, lambda: _ridge_med(alpha=40.0)),
        ("v3 28d (Shallow GBR)", v3_28d, lambda: _gbr_shallow(min_leaf=20, n_estimators=50)),
    ]:
        cv = walk_forward_cv(weekly, feats, "target_28d", fn, **wf_weekly,
                             purge=4, halflife_days=RECENCY_HALFLIFE_DAYS["weekly"])
        print(f"{'28d':<8} {label:<26} {len(feats):>6} {cv['mape']:>8.4f} {cv['naive_mape']:>8.4f} "
              f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} {cv['folds']:>6}")

    monthly["target_90d"] = make_return_target(monthly, 3)
    v3_90d = available(monthly, FEATS_90D)
    v3_90d_t4 = available(monthly, FEATS_90D + T4_FEATURES)
    print()
    for label, feats in [
        ("v3 90d (BayesianRidge)", v3_90d),
        ("v3 90d +T4 (Bayesian)", v3_90d_t4),
    ]:
        cv = walk_forward_cv(monthly, feats, "target_90d", _bayesian_90d, **wf_monthly,
                             purge=3, halflife_days=RECENCY_HALFLIFE_DAYS["monthly"])
        print(f"{'90d':<8} {label:<26} {len(feats):>6} {cv['mape']:>8.4f} {cv['naive_mape']:>8.4f} "
              f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} {cv['folds']:>6}")

    # Regime classifier (leak-free walk-forward with LogisticRegression Pipeline)
    price_ref = "adj_price" if "adj_price" in monthly.columns else "avg_price"
    monthly["fwd_6m_ret"] = (monthly[price_ref].shift(-6) / monthly[price_ref] - 1.0) * 100.0
    monthly["bear_signal"] = (monthly["fwd_6m_ret"] < -10.0).astype(float)
    monthly.loc[monthly["fwd_6m_ret"].isna(), "bear_signal"] = np.nan

    reg_feats = available(monthly, FEATS_REGIME)
    df_reg = monthly.dropna(subset=["bear_signal"]).reset_index(drop=True)
    X_all = df_reg[reg_feats].to_numpy(dtype=float)
    y_all = df_reg["bear_signal"].to_numpy(dtype=int)

    preds, actuals = [], []
    i = wf_monthly["min_train"]
    while i + wf_monthly["eval_win"] <= len(df_reg):
        train_end = i - 6
        if train_end >= 24 and len(np.unique(y_all[:train_end])) >= 2:
            m = _logreg_regime()
            m.fit(X_all[:train_end], y_all[:train_end])
            preds.extend(m.predict(X_all[i:i + wf_monthly["eval_win"]]))
            actuals.extend(y_all[i:i + wf_monthly["eval_win"]])
        i += wf_monthly["step"]

    if preds:
        acc = float(np.mean(np.array(preds) == np.array(actuals)))
        print(f"\nRegime (L2 LogisticRegression, {len(reg_feats)} feats): walk-forward accuracy = {acc:.3f} (n={len(preds)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
