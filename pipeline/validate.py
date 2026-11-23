"""Validate predictions against actual prices and update Online ACI state."""

import logging

import pandas as pd

from pipeline.config import (
    CONFORMAL_ACI_GAMMA,
    CONFORMAL_ALPHA,
    MODEL_VERSION,
    TARGET_MATCH_TOLERANCE_DAYS,
)
from pipeline.conformal import update_aci_alpha
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def backfill_naive_baseline(conn, prices) -> int:
    """Fill the naive benchmark on rows validated before v2.3 recorded it."""
    stale = pd.read_sql(
        """SELECT v.id, v.date, v.actual_price, f.forecast_date
           FROM validation_log v
           JOIN forecast_ledger f
             ON f.target_date = v.date AND f.horizon_days = v.horizon_days
           WHERE v.anchor_price IS NULL AND v.actual_price IS NOT NULL""",
        conn,
    )
    if stale.empty:
        return 0

    stale["_run"] = pd.to_datetime(stale["forecast_date"])
    anchored = pd.merge_asof(
        stale.sort_values("_run"),
        prices.rename(columns={"date": "_run", "avg_price": "anchor_price"}),
        on="_run", direction="backward",
    ).dropna(subset=["anchor_price"])
    if anchored.empty:
        return 0

    updates = []
    for _, r in anchored.iterrows():
        actual, anchor = float(r["actual_price"]), float(r["anchor_price"])
        naive_abs = abs(anchor - actual)
        updates.append((
            anchor, naive_abs, naive_abs / actual if actual else 0.0,
            r["date"], int(r["id"]),
        ))
    conn.executemany(
        """UPDATE validation_log
           SET anchor_price = ?, naive_abs_error = ?, naive_pct_error = ?,
               actual_date = COALESCE(actual_date, ?)
           WHERE id = ?""",
        updates,
    )
    conn.commit()
    return len(updates)


def _update_online_aci_state(conn, settled_rows: list[dict]) -> None:
    """Apply Online Adaptive Conformal Inference updates per horizon from newly settled rows.

    Only forecasts made by the current MODEL_VERSION feed the state: a hit or
    miss against another version's bands says nothing about how the current
    bands are calibrated (e.g. v2.4-era forecasts settling after a v3 deploy).
    """
    by_h: dict[int, list[int]] = {}
    for r in settled_rows:
        if r.get("in_interval") is None or r.get("model_version") != MODEL_VERSION:
            continue
        by_h.setdefault(int(r["horizon_days"]), []).append(int(r["in_interval"]))

    if not by_h:
        return

    for h, hits in by_h.items():
        row = conn.execute(
            "SELECT alpha_t, n_updates FROM conformal_state WHERE horizon_days = ?",
            (h,),
        ).fetchone()
        alpha_t = float(row[0]) if row else float(CONFORMAL_ALPHA)
        n_updates = int(row[1]) if row else 0

        for hit in hits:
            alpha_t = update_aci_alpha(
                alpha_t,
                in_interval=bool(hit),
                target_alpha=CONFORMAL_ALPHA,
                gamma=CONFORMAL_ACI_GAMMA,
            )
            n_updates += 1

        # Compute trailing empirical coverage from validation_log for reporting
        cov_row = conn.execute(
            """SELECT AVG(in_interval) FROM (
                 SELECT in_interval FROM validation_log
                 WHERE horizon_days = ? AND in_interval IS NOT NULL
                 ORDER BY date DESC LIMIT 60
               )""",
            (h,),
        ).fetchone()
        emp_cov = float(cov_row[0]) if (cov_row and cov_row[0] is not None) else None

        conn.execute(
            """INSERT OR REPLACE INTO conformal_state
               (horizon_days, alpha_t, n_updates, empirical_coverage, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (h, round(alpha_t, 5), n_updates, round(emp_cov, 4) if emp_cov is not None else None),
        )
    conn.commit()


def validate_predictions(
    today: str,
    tolerance_days: int = TARGET_MATCH_TOLERANCE_DAYS,
) -> list[dict]:
    """Score every matured forecast, record interval coverage, and update Online ACI."""
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

    filled = backfill_naive_baseline(conn, prices)
    if filled:
        log.info(f"Backfilled the naive benchmark on {filled} historical validation rows")

    pending = pd.read_sql(
        """SELECT f.forecast_date, f.target_date, f.horizon_days,
                  f.predicted_price, f.lower_bound, f.upper_bound,
                  f.model_version
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

    settle_src = prices.rename(columns={
        "date": "_target",
        "avg_price": "actual_price",
    }).copy()
    settle_src["actual_date"] = settle_src["_target"]
    df = pd.merge_asof(
        pending.sort_values("_target"), settle_src,
        on="_target", direction="forward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    anchor_src = prices.rename(columns={
        "date": "_run",
        "avg_price": "anchor_price",
    })
    df = pd.merge_asof(
        df.sort_values("_run"), anchor_src, on="_run", direction="backward",
    )

    unsettled = int(df["actual_price"].isna().sum())
    df = df.dropna(subset=["actual_price"]).sort_values("_target")
    if df.empty:
        log.info(
            f"No matured forecasts could be settled ({unsettled} still awaiting an auction)"
        )
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

        lb = row.get("lower_bound")
        ub = row.get("upper_bound")
        if pd.notna(lb) and pd.notna(ub):
            lb_f, ub_f = float(lb), float(ub)
            in_interval = 1 if (lb_f <= actual <= ub_f) else 0
        else:
            lb_f = ub_f = in_interval = None

        actual_date = row["actual_date"].strftime("%Y-%m-%d")
        if actual_date != row["target_date"]:
            carried += 1

        conn.execute(
            """INSERT OR REPLACE INTO validation_log
               (date, horizon_days, predicted_price, actual_price,
                abs_error, pct_error, actual_date, anchor_price,
                naive_abs_error, naive_pct_error,
                lower_bound, upper_bound, in_interval)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["target_date"], int(row["horizon_days"]), predicted, actual,
                abs_error, pct_error, actual_date,
                float(anchor) if anchor is not None else None,
                naive_abs, naive_pct,
                lb_f, ub_f, in_interval,
            ),
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
            "in_interval": in_interval,
            "model_version": row.get("model_version"),
        })

    conn.commit()
    _update_online_aci_state(conn, results)
    conn.close()

    scored = [r for r in results if r["naive_abs_error"] is not None]
    if scored:
        wins = sum(r["beat_naive"] for r in scored)
        log.info(
            f"Validated {len(results)} forecasts ({carried} settled on a later auction, "
            f"{unsettled} still awaiting one) — beat naive in {wins}/{len(scored)}"
        )
    else:
        log.info(f"Validated {len(results)} forecasts ({carried} settled on a later auction)")
    return results
