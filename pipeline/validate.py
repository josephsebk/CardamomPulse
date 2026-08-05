"""Validate predictions against actual prices."""

import logging

import pandas as pd

from pipeline.config import TARGET_MATCH_TOLERANCE_DAYS
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def validate_predictions(today: str,
                         tolerance_days: int = TARGET_MATCH_TOLERANCE_DAYS
                         ) -> list[dict]:
    """Score every matured forecast that has not been validated yet.

    Target dates are calendar dates, but auctions run ~5.14 days a week, so
    ~41% of them never host an auction (55% at the 1-day horizon). Settling
    only on an exact date match left those forecasts permanently unscored and
    made the published track record a non-random subsample of the ledger.
    Each forecast is therefore settled against the first auction at or after
    its target_date, within tolerance_days — the same rule that defines the
    daily training target.

    Scanning the whole ledger rather than just today's target dates also
    backfills forecasts that an earlier run skipped.

    Every row stores the naive "no change" benchmark alongside the model:
    forecasts are built as anchor*exp(return), so the anchor is exactly the
    random-walk forecast and is the baseline the model has to beat.
    """
    conn = get_conn()

    prices = pd.read_sql(
        "SELECT date, avg_price FROM auction_daily "
        "WHERE avg_price IS NOT NULL AND date <= ? ORDER BY date",
        conn, params=(today,),
    )
    if prices.empty:
        log.info("No auction prices available — skipping validation")
        conn.close()
        return []
    prices["date"] = pd.to_datetime(prices["date"])

    pending = pd.read_sql(
        """SELECT f.forecast_date, f.target_date, f.horizon_days,
                  f.predicted_price
           FROM forecast_ledger f
           LEFT JOIN validation_log v
             ON v.date = f.target_date AND v.horizon_days = f.horizon_days
           WHERE f.target_date <= ? AND v.id IS NULL
           ORDER BY f.target_date""",
        conn, params=(today,),
    )
    if pending.empty:
        log.info("No matured forecasts awaiting validation")
        conn.close()
        return []

    pending["_target"] = pd.to_datetime(pending["target_date"])
    pending["_run"] = pd.to_datetime(pending["forecast_date"])

    # Settle on the first auction at or after target_date (exact dates match
    # as-is, so previously-validated rows keep the same actual).
    settle_src = prices.rename(columns={"date": "_target",
                                        "avg_price": "actual_price"}).copy()
    settle_src["actual_date"] = settle_src["_target"]
    df = pd.merge_asof(
        pending.sort_values("_target"), settle_src,
        on="_target", direction="forward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    # Anchor = last auction at or before the forecast was made. This is the
    # price the model itself anchored on, so it is the honest random walk.
    anchor_src = prices.rename(columns={"date": "_run",
                                        "avg_price": "anchor_price"})
    df = pd.merge_asof(
        df.sort_values("_run"), anchor_src, on="_run", direction="backward",
    )

    unsettled = int(df["actual_price"].isna().sum())
    df = df.dropna(subset=["actual_price"])
    if df.empty:
        log.info(f"No matured forecasts could be settled "
                 f"({unsettled} still awaiting an auction)")
        conn.close()
        return []

    results = []
    carried = 0
    for _, row in df.iterrows():
        predicted = float(row["predicted_price"])
        actual = float(row["actual_price"])
        abs_error = abs(predicted - actual)
        pct_error = abs_error / actual if actual else 0.0

        anchor = row.get("anchor_price")
        if pd.notna(anchor):
            naive_abs = abs(float(anchor) - actual)
            naive_pct = naive_abs / actual if actual else 0.0
        else:
            anchor = naive_abs = naive_pct = None

        actual_date = row["actual_date"].strftime("%Y-%m-%d")
        if actual_date != row["target_date"]:
            carried += 1

        conn.execute(
            """INSERT OR REPLACE INTO validation_log
               (date, horizon_days, predicted_price, actual_price,
                abs_error, pct_error, actual_date, anchor_price,
                naive_abs_error, naive_pct_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row["target_date"], int(row["horizon_days"]), predicted, actual,
             abs_error, pct_error, actual_date,
             float(anchor) if anchor is not None else None,
             naive_abs, naive_pct),
        )
        results.append({
            "date": row["target_date"],
            "actual_date": actual_date,
            "horizon_days": int(row["horizon_days"]),
            "predicted": predicted,
            "actual": actual,
            "abs_error": round(abs_error, 1),
            "pct_error": round(pct_error, 4),
            "naive_abs_error": round(naive_abs, 1) if naive_abs is not None else None,
            "beat_naive": (naive_abs is not None and abs_error < naive_abs),
        })

    conn.commit()
    conn.close()

    scored = [r for r in results if r["naive_abs_error"] is not None]
    if scored:
        wins = sum(r["beat_naive"] for r in scored)
        log.info(f"Validated {len(results)} forecasts "
                 f"({carried} settled on a later auction, {unsettled} still "
                 f"awaiting one) — beat the naive baseline in {wins}/"
                 f"{len(scored)}")
    else:
        log.info(f"Validated {len(results)} forecasts "
                 f"({carried} settled on a later auction)")
    return results
