#!/usr/bin/env python3
"""
Backtest: do mean-reversion terms produce a model that beats a random walk?

Motivation. The auction series is not a random walk — lag-1 return
autocorrelation is about -0.13 (~7 SE, n=3150) and variance ratios run
0.69-0.87 across q=2..60, i.e. persistent multi-scale mean reversion. That
is genuine and not a composition artifact: rebuilding the index with
auctioneer fixed effects (21 auctioneers, effects spanning 6.1%) makes the
reversion *stronger*, not weaker, which is the opposite of what a
lot-composition story predicts.

The open question is whether that unconditional structure can be turned into
a forecast that beats "assume no change". An earlier single-window test
(fit pre-2026, scored on Feb-Aug 2026) suggested yes. It does not survive
walk-forward validation across the full history and all regimes — that window
was one regime, with n=25-35 at the long horizons.

This script runs the honest version: walk-forward CV over 2014-2026 with
calendar-aligned targets, scoring every feature set against the naive
baseline on its own evaluation rows.

Needs only the local auction XLS — no network.

Usage:
  python3 backtest_mean_reversion.py            # 1, 7, 14, 28, 90 day horizons
  python3 backtest_mean_reversion.py --quick    # skip the slow daily GBR runs
"""

import argparse

import numpy as np
from sklearn.linear_model import Ridge

from pipeline.assemble import resample_monthly, resample_weekly
from pipeline.collectors.auction import aggregate_daily, load_xls_fallback
from pipeline.config import WF_CONFIG
from pipeline.features import add_tier1
from pipeline.models import (
    _bayesian_90d, _gbr_28d, _gbr_7d, _gbr_short,
    make_return_target, make_return_target_calendar, walk_forward_cv,
)


def add_reversion_terms(df, windows):
    """Deviation of the CURRENT price from its own trailing mean.

    T1 computes every feature on price.shift(1), so the models never see the
    price the forecast is actually anchored on — they are asked to predict the
    move starting from a level they cannot observe. Today's auction has closed
    by the 6PM IST run and predict_all already anchors on it, so using it here
    is causal.
    """
    lp = np.log(df["avg_price"])
    names = []
    for w in windows:
        col = f"dev_{w}"
        df[col] = lp - lp.rolling(w, min_periods=max(2, w // 2)).mean()
        names.append(col)
    df["ret_1"] = lp.diff()
    names.append("ret_1")
    return names


def usable(df, feats, floor):
    return [c for c in feats if df[c].notna().sum() > floor]


def report(label, df, feats, target_col, model_fn, cfg, purge, anchor="avg_price"):
    cv = walk_forward_cv(df, feats, target_col, model_fn, **cfg, purge=purge,
                         anchor_col=anchor)
    print(f"  {label:<30} {cv['mape']:>7.4f} {cv['naive_mape']:>7.4f} "
          f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} "
          f"{cv.get('dir_acc', float('nan')):>6.2f} {cv['folds']:>6}")
    return cv


def main(quick=False):
    daily = aggregate_daily(load_xls_fallback())
    weekly, monthly = resample_weekly(daily), resample_monthly(daily)

    t1_d = usable(daily, add_tier1(daily), 500)
    mr_d = add_reversion_terms(daily, [5, 10, 20, 60])
    daily["zscore_20"] = daily["dev_20"] / np.log(
        daily["avg_price"]).diff().rolling(20, min_periods=10).std().clip(lower=1e-6)
    mr_d.append("zscore_20")

    t1_w = usable(weekly, add_tier1(weekly), 100)
    mr_w = add_reversion_terms(weekly, [3, 6, 12])
    t1_m = usable(monthly, add_tier1(monthly), 25)
    mr_m = add_reversion_terms(monthly, [3, 6, 12])

    print(f"daily {daily.shape} | weekly {weekly.shape} | monthly {monthly.shape}")
    print("\nWalk-forward CV. skill = 1 - MAE/MAE_naive; >0 beats the random walk.")
    print("Each row is scored against the naive baseline on its own eval rows,")
    print("so skill is comparable across sets even where MAPE is not.\n")
    header = (f"  {'feature set':<30} {'MAPE':>7} {'naive':>7} {'skill':>8} "
              f"{'TheilU':>7} {'dir':>6} {'folds':>6}")

    for h in ([7] if quick else [1, 7, 14]):
        print(f"{h}-day"); print(header)
        daily[f"t{h}"] = make_return_target_calendar(daily, h)
        gbr = (lambda _h=h: _gbr_short(_h)) if h < 7 else _gbr_7d
        report("T1 only (current design)", daily, t1_d, f"t{h}", gbr,
               WF_CONFIG["daily"], h)
        report("T1 + reversion", daily, t1_d + mr_d, f"t{h}", gbr,
               WF_CONFIG["daily"], h)
        report("reversion only (GBR)", daily, mr_d, f"t{h}", gbr,
               WF_CONFIG["daily"], h)
        report("reversion only (ridge)", daily, mr_d, f"t{h}",
               lambda: Ridge(alpha=1.0), WF_CONFIG["daily"], h)
        print()

    weekly["t28"] = make_return_target(weekly, 4)   # calendar-resampled frame
    print("28-day"); print(header)
    report("T1 only (current design)", weekly, t1_w, "t28", _gbr_28d,
           WF_CONFIG["weekly"], 4)
    report("T1 + reversion", weekly, t1_w + mr_w, "t28", _gbr_28d,
           WF_CONFIG["weekly"], 4)
    report("reversion only (ridge)", weekly, mr_w, "t28",
           lambda: Ridge(alpha=1.0), WF_CONFIG["weekly"], 4)

    monthly["t90"] = make_return_target(monthly, 3)
    print("\n90-day"); print(header)
    report("T1 only (current design)", monthly, t1_m, "t90", _bayesian_90d,
           WF_CONFIG["monthly"], 3)
    report("T1 + reversion", monthly, t1_m + mr_m, "t90", _bayesian_90d,
           WF_CONFIG["monthly"], 3)
    report("reversion only (bayes)", monthly, mr_m, "t90", _bayesian_90d,
           WF_CONFIG["monthly"], 3)
    report("reversion only (ridge)", monthly, mr_m, "t90",
           lambda: Ridge(alpha=1.0), WF_CONFIG["monthly"], 3)

    print("\nRead the skill column. On the full history no configuration beats the")
    print("naive baseline; the simplest ones merely match it, while the")
    print("feature-heavy models are far worse. Reversion is real in the series")
    print("but does not survive as a tradeable forecast edge.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="only the 7-day daily horizon (the daily GBR runs are slow)")
    main(**vars(p.parse_args()))
