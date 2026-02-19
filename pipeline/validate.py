"""Validate predictions against actual prices."""

import logging
import pandas as pd
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def validate_predictions(today: str) -> list[dict]:
    """Compare past predictions against today's actual price.

    Looks at forecast_ledger entries whose target_date is today,
    and compares with the actual price from auction_daily.
    """
    conn = get_conn()

    # Get today's actual price
    cur = conn.execute(
        "SELECT avg_price FROM auction_daily WHERE date = ?", (today,)
    )
    row = cur.fetchone()
    if not row or row[0] is None:
        log.info(f"No actual price for {today}, skipping validation")
        conn.close()
        return []

    actual_price = row[0]

    # Find predictions targeting today
    preds = pd.read_sql(
        "SELECT * FROM forecast_ledger WHERE target_date = ?",
        conn, params=(today,),
    )

    if preds.empty:
        log.info(f"No predictions targeting {today}")
        conn.close()
        return []

    results = []
    for _, pred in preds.iterrows():
        predicted = pred["predicted_price"]
        abs_error = abs(predicted - actual_price)
        pct_error = abs_error / actual_price if actual_price else 0

        # Insert into validation log
        conn.execute(
            """INSERT OR REPLACE INTO validation_log
               (date, horizon_days, predicted_price, actual_price, abs_error, pct_error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (today, pred["horizon_days"], predicted, actual_price, abs_error, pct_error),
        )
        results.append({
            "date": today,
            "horizon_days": int(pred["horizon_days"]),
            "predicted": predicted,
            "actual": actual_price,
            "abs_error": round(abs_error, 1),
            "pct_error": round(pct_error, 4),
        })
        log.info(
            f"Validated {pred['horizon_days']}d prediction: "
            f"pred=₹{predicted:.0f} actual=₹{actual_price:.0f} "
            f"err={pct_error:.1%}"
        )

    conn.commit()
    conn.close()
    return results
