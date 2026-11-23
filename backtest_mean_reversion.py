#!/usr/bin/env python3
"""
Backtest: evaluate v3.0 causal house-mix-adjusted index, leak-free regularized
models, and shrinkage against the naive random-walk baseline.

Walk-forward CV over 2014-2026 with calendar-aligned targets, scoring every
model/feature configuration against the naive persistence baseline on the exact
same evaluation rows.

Needs only the local auction XLS — no network.

Usage:
  python3 backtest_mean_reversion.py            # 1, 7, 14, 28, 90 day horizons
  python3 backtest_mean_reversion.py --quick    # only 1d and 7d horizons
"""

import argparse

import numpy as np

from pipeline.assemble import resample_monthly, resample_weekly
from pipeline.collectors.auction import aggregate_daily, load_xls_fallback
from pipeline.config import RECENCY_HALFLIFE_DAYS, WF_CONFIG
from pipeline.features import add_tier1
from pipeline.models import (
    _bayesian_90d,
    _gbr_shallow,
    _huber_short,
    _ridge_med,
    _ridge_short,
    make_return_target,
    make_return_target_calendar,
    walk_forward_cv,
)


def usable(df, feats, floor):
    return [c for c in feats if c in df.columns and df[c].notna().sum() > floor]


def report(label, df, feats, target_col, model_fn, cfg, purge, anchor="adj_price", halflife=None):
    cv = walk_forward_cv(
        df, feats, target_col, model_fn, **cfg,
        purge=purge, anchor_col=anchor, halflife_days=halflife,
    )
    print(f"  {label:<34} {cv['mape']:>7.4f} {cv['naive_mape']:>7.4f} "
          f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} "
          f"{cv.get('dir_acc', float('nan')):>6.2f} {cv['folds']:>6}")
    return cv


def main(quick=False):
    daily = aggregate_daily(load_xls_fallback())
    weekly, monthly = resample_weekly(daily), resample_monthly(daily)

    t1_d = usable(daily, add_tier1(daily, freq="daily"), 500)
    t1_w = usable(weekly, add_tier1(weekly, freq="weekly"), 100)
    t1_m = usable(monthly, add_tier1(monthly, freq="monthly"), 25)

    mr_cols = [c for c in ["ret_1", "dev_ma_1", "dev_ma_2", "dev_ma_3", "zscore_ma3", "house_effect_log"] if c in daily.columns]
    mr_w = [c for c in ["ret_1", "dev_ma_1", "dev_ma_2", "dev_ma_3"] if c in weekly.columns]
    mr_m = [c for c in ["ret_1", "dev_ma_1", "dev_ma_2", "dev_ma_3"] if c in monthly.columns]

    print(f"daily {daily.shape} | weekly {weekly.shape} | monthly {monthly.shape}")
    print("\nWalk-forward CV (v3.0 leak-free Pipeline + house-mix-adjusted index).")
    print("skill = 1 - MAE/MAE_naive; >0 beats the random walk.\n")
    header = (f"  {'configuration':<34} {'MAPE':>7} {'naive':>7} {'skill':>8} "
              f"{'TheilU':>7} {'dir':>6} {'folds':>6}")

    for h in ([1, 7] if quick else [1, 7, 14]):
        print(f"{h}-day"); print(header)
        daily[f"t{h}_raw"] = make_return_target_calendar(daily, h, price_col="avg_price")
        daily[f"t{h}_adj"] = make_return_target_calendar(daily, h, price_col="adj_price")
        hl = RECENCY_HALFLIFE_DAYS["daily"]

        report("T1 Huber (raw avg_price)", daily, t1_d, f"t{h}_raw",
               lambda _h=h: _huber_short(_h), WF_CONFIG["daily"], h, anchor="avg_price", halflife=hl)
        report("T1 Huber (adj_price)", daily, t1_d, f"t{h}_adj",
               lambda _h=h: _huber_short(_h), WF_CONFIG["daily"], h, anchor="adj_price", halflife=hl)
        report("T1 Ridge (adj_price)", daily, t1_d, f"t{h}_adj",
               lambda _h=h: _ridge_short(_h), WF_CONFIG["daily"], h, anchor="adj_price", halflife=hl)
        report("Reversion+House Ridge (adj_price)", daily, mr_cols, f"t{h}_adj",
               lambda _h=h: _ridge_short(_h), WF_CONFIG["daily"], h, anchor="adj_price", halflife=hl)
        if h == 14:
            report("T1 Shallow GBR (adj_price)", daily, t1_d, f"t{h}_adj",
                   lambda: _gbr_shallow(min_leaf=30, n_estimators=60),
                   WF_CONFIG["daily"], h, anchor="adj_price", halflife=hl)
        print()

    if not quick:
        weekly["t28"] = make_return_target(weekly, 4, price_col="adj_price")
        print("28-day"); print(header)
        hl28 = RECENCY_HALFLIFE_DAYS["weekly"]
        report("T1 Ridge (adj_price)", weekly, t1_w, "t28",
               lambda: _ridge_med(alpha=40.0), WF_CONFIG["weekly"], 4, anchor="adj_price", halflife=hl28)
        report("T1 Shallow GBR (adj_price)", weekly, t1_w, "t28",
               lambda: _gbr_shallow(min_leaf=20, n_estimators=50),
               WF_CONFIG["weekly"], 4, anchor="adj_price", halflife=hl28)
        report("Reversion Ridge (adj_price)", weekly, mr_w, "t28",
               lambda: _ridge_med(alpha=20.0), WF_CONFIG["weekly"], 4, anchor="adj_price", halflife=hl28)

        monthly["t90"] = make_return_target(monthly, 3, price_col="adj_price")
        print("\n90-day"); print(header)
        hl90 = RECENCY_HALFLIFE_DAYS["monthly"]
        report("T1 BayesianRidge (adj_price)", monthly, t1_m, "t90",
               _bayesian_90d, WF_CONFIG["monthly"], 3, anchor="adj_price", halflife=hl90)
        report("Reversion BayesianRidge", monthly, mr_m, "t90",
               _bayesian_90d, WF_CONFIG["monthly"], 3, anchor="adj_price", halflife=hl90)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="only run 1d and 7d horizons")
    main(**vars(p.parse_args()))
