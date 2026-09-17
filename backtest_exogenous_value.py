#!/usr/bin/env python3
"""
Backtest: does an exogenous data source add forecasting value?

The bar any new input has to clear is not "does it beat the current model" —
the current models lose to a naive random walk at every horizon. It is
"does it beat the naive baseline". This script measures that.

Two traps it is built to avoid:

1. Different feature sets drop different rows. walk_forward_cv dropna()s on
   its own feature list, so a set with long warm-ups is silently scored on a
   different, shorter sample than a set without. Comparing those numbers is
   meaningless. Everything here is scored on one common sample.

2. Long warm-up windows are expensive. The existing T3 set includes
   rain_cum_182 and enso_lag12m; requiring those non-NaN cuts usable weekly
   rows from 546 to 177 and monthly rows from 130 to 82. On the monthly frame
   that is most of the training data, spent before the feature has shown any
   value. Prefer short aggregation windows (28-56 day) for anything new.

Result for the weather/ENSO data the project already collects (the closest
available proxy for satellite crop data — same class of slow exogenous
signal):

  28-day  price only -0.591 | + weather -0.479 | weather only (ridge) -0.394
  90-day  price only -0.331 | + weather -0.821 | weather only (ridge) -2.258

Nothing beats the naive baseline. The 28-day direction is positive but flips
sign when the common-sample constraint is removed, on 5 folds, so it is not a
reliable effect. The 90-day direction is clearly negative.

Needs no network: reads the committed archive.csv and external_*.csv.

Usage:
  python3 backtest_exogenous_value.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pipeline.assemble import resample_monthly, resample_weekly
from pipeline.config import WF_CONFIG
from pipeline.features import add_tier1, add_tier3
from pipeline.models import (
    _bayesian_90d, _gbr_28d, make_return_target, walk_forward_cv,
)

ROOT = "."
SEASON_TO_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                   "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def build_daily():
    """Price series plus the exogenous data already committed to the repo."""
    a = pd.read_csv(f"{ROOT}/cardamom_webapp/data/archive.csv")
    d = (a.dropna(subset=["actual_avg_price_inr_per_kg"])
         .drop_duplicates(subset=["date"])[["date", "actual_avg_price_inr_per_kg"]]
         .rename(columns={"actual_avg_price_inr_per_kg": "avg_price"})
         .sort_values("date").reset_index(drop=True))
    d["date"] = pd.to_datetime(d["date"])
    # archive.csv carries no max_price; it feeds only max_avg_spread, and T1 is
    # held constant across every comparison, so a placeholder cannot bias them.
    d["max_price"] = d["avg_price"]

    for path, drop in [(f"{ROOT}/external_idukki_weather.csv", "Date"),
                       (f"{ROOT}/external_guatemala_weather.csv", "Date")]:
        w = pd.read_csv(path)
        w["date"] = pd.to_datetime(w[drop])
        d = d.merge(w.drop(columns=[drop]), on="date", how="left")

    e = pd.read_csv(f"{ROOT}/external_enso_oni.csv")
    e["month"] = e["SEAS"].map(SEASON_TO_MONTH)
    e["edate"] = pd.to_datetime(
        e["YR"].astype(str) + "-" + e["month"].astype(str) + "-15", errors="coerce")
    e = e.dropna(subset=["edate"]).sort_values("edate")
    daily_enso = e.set_index("edate")[["ANOM"]].resample("D").ffill().reset_index()
    daily_enso.columns = ["date", "ENSO"]
    d = pd.merge_asof(d.sort_values("date"), daily_enso, on="date", direction="backward")

    # Truncate to where the exogenous data actually exists, so the comparison
    # is not partly a comparison of different date ranges.
    end = d.loc[d["rain_mm"].notna(), "date"].max()
    return d[d["date"] <= end].reset_index(drop=True)


def compare(df, base_feats, new_feats, horizon_rows, model_fn, cfg, purge, label):
    """Score base vs base+new vs new-only on one identical sample."""
    df = df.copy()
    df["_target"] = make_return_target(df, horizon_rows)
    all_feats = list(dict.fromkeys(base_feats + new_feats))
    common = df.dropna(subset=all_feats + ["_target", "avg_price"]).reset_index(drop=True)

    print(f"\n{label} — common usable rows: {len(common)}")
    print(f"  {'feature set':<30} {'MAPE':>7} {'naive':>7} {'skill':>8} "
          f"{'TheilU':>7} {'dir':>6} {'folds':>6}")
    results = {}
    for name, feats, mfn in [
        ("base (price only)", base_feats, model_fn),
        ("base + new", base_feats + new_feats, model_fn),
        ("new only", new_feats, model_fn),
        ("new only (ridge)", new_feats, lambda: Ridge(alpha=1.0)),
    ]:
        cv = walk_forward_cv(common, feats, "_target", mfn, **cfg,
                             purge=purge, anchor_col="avg_price")
        if np.isnan(cv["mape"]):
            print(f"  {name:<30} (too few rows for this CV config)")
            continue
        print(f"  {name:<30} {cv['mape']:>7.4f} {cv['naive_mape']:>7.4f} "
              f"{cv['skill']:>+8.3f} {cv['theil_u']:>7.3f} "
              f"{cv.get('dir_acc', float('nan')):>6.2f} {cv['folds']:>6}")
        results[name] = cv

    if "base (price only)" in results and "base + new" in results:
        delta = results["base + new"]["skill"] - results["base (price only)"]["skill"]
        beats = any(r["skill"] > 0 for r in results.values())
        print(f"  --> adds {delta:+.3f} skill ({'helps' if delta > 0 else 'hurts'}); "
              f"beats the naive baseline: {'YES' if beats else 'NO'}")
    return results


def main():
    daily = build_daily()
    print(f"daily rows {len(daily)}, through {daily['date'].max().date()}")
    weekly, monthly = resample_weekly(daily), resample_monthly(daily)

    def split(df, floor, is_monthly=False):
        t1 = [c for c in add_tier1(df) if df[c].notna().sum() > floor]
        t3 = [c for c in add_tier3(df, is_monthly=is_monthly) if df[c].notna().sum() > floor]
        return t1, t3

    t1w, t3w = split(weekly, 100)
    t1m, t3m = split(monthly, 25, is_monthly=True)

    compare(weekly, t1w, t3w, 4, _gbr_28d, WF_CONFIG["weekly"], 4,
            "28-day (weekly frame), new = weather/ENSO")
    compare(monthly, t1m, t3m, 3, _bayesian_90d, WF_CONFIG["monthly"], 3,
            "90-day (monthly frame), new = weather/ENSO")

    print("\nTo evaluate a different source, merge it into build_daily() and pass "
          "its feature names as new_feats. The bar is skill > 0, not merely "
          "beating the current model.")


if __name__ == "__main__":
    main()
