"""Export pipeline results to static JSON files for the web frontend."""

import json
import logging
from datetime import datetime

import pandas as pd
from pipeline.config import EXPORT_DIR, MODEL_VERSION, ensure_dirs
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def export_json():
    """Generate all JSON files consumed by the web dashboard."""
    ensure_dirs()
    conn = get_conn()

    # ── 1. dashboard.json — main dashboard payload ───────────────────
    # Latest price
    latest = pd.read_sql(
        "SELECT * FROM auction_daily ORDER BY date DESC LIMIT 30", conn
    )
    if latest.empty:
        log.warning("No auction data to export")
        conn.close()
        return

    spot = latest.iloc[0]
    prev = latest.iloc[1] if len(latest) > 1 else spot
    daily_change = spot["avg_price"] - prev["avg_price"]
    daily_change_pct = daily_change / prev["avg_price"] * 100 if prev["avg_price"] else 0

    # Latest forecasts
    forecasts = pd.read_sql(
        """SELECT * FROM forecast_ledger
           WHERE forecast_date = (SELECT MAX(forecast_date) FROM forecast_ledger)
           ORDER BY horizon_days""",
        conn,
    )

    forecast_list = []
    for _, f in forecasts.iterrows():
        entry = {
            "horizon_days": int(f["horizon_days"]),
            "target_date": f["target_date"],
            "predicted_price": f["predicted_price"],
            "model_version": f["model_version"],
        }
        if pd.notna(f.get("lower_bound")):
            entry["lower_bound"] = f["lower_bound"]
            entry["upper_bound"] = f["upper_bound"]
        forecast_list.append(entry)

    # Latest regime
    regime_row = pd.read_sql(
        "SELECT * FROM regime_ledger ORDER BY forecast_date DESC LIMIT 1", conn
    )
    regime = None
    if not regime_row.empty:
        r = regime_row.iloc[0]
        regime = {
            "bear_probability": r["bear_probability"],
            "label": r["regime_label"],
            "forecast_date": r["forecast_date"],
        }

    dashboard = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "spot_price": {
            "date": spot["date"],
            "avg_price": spot["avg_price"],
            "max_price": spot["max_price"],
            "daily_change": round(daily_change, 1),
            "daily_change_pct": round(daily_change_pct, 2),
            "volume_kg": spot["sold_kg"],
        },
        "forecasts": forecast_list,
        "regime": regime,
        "model_version": MODEL_VERSION,
    }
    _write_json(EXPORT_DIR / "dashboard.json", dashboard)

    # ── 2. price_history.json — last 90 days ─────────────────────────
    history = pd.read_sql(
        "SELECT date, avg_price, max_price, sold_kg FROM auction_daily "
        "ORDER BY date DESC LIMIT 90",
        conn,
    )
    history = history.sort_values("date")
    history_list = history.to_dict(orient="records")
    _write_json(EXPORT_DIR / "price_history.json", {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "prices": history_list,
    })

    # ── 3. track_record.json — validation metrics ────────────────────
    validations = pd.read_sql(
        "SELECT * FROM validation_log ORDER BY date DESC LIMIT 200", conn
    )
    if not validations.empty:
        # Compute metrics by horizon
        metrics_by_horizon = {}
        for horizon in validations["horizon_days"].unique():
            subset = validations[validations["horizon_days"] == horizon]
            recent_30 = subset.head(30)
            metrics_by_horizon[str(int(horizon))] = {
                "n": len(recent_30),
                "mape": round(float(recent_30["pct_error"].mean()), 4),
                "mae": round(float(recent_30["abs_error"].mean()), 1),
                "rmse": round(float((recent_30["abs_error"] ** 2).mean() ** 0.5), 1),
                "hit_5pct": round(
                    float((recent_30["pct_error"] <= 0.05).mean()), 3
                ),
            }

        track_record = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "total_predictions": len(validations),
            "metrics_by_horizon": metrics_by_horizon,
            "recent_validations": validations.head(30).to_dict(orient="records"),
        }
    else:
        track_record = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "total_predictions": 0,
            "metrics_by_horizon": {},
            "recent_validations": [],
        }
    _write_json(EXPORT_DIR / "track_record.json", track_record)

    # ── 4. archive.csv — for backward compatibility with existing UI ──
    archive = pd.read_sql(
        """SELECT a.date,
                  a.avg_price AS actual_avg_price_inr_per_kg,
                  f.predicted_price AS predicted_avg_price_inr_per_kg,
                  f.forecast_date AS model_run_date,
                  f.horizon_days,
                  f.model_version
           FROM auction_daily a
           LEFT JOIN forecast_ledger f
             ON a.date = f.target_date AND f.horizon_days = 7
           ORDER BY a.date""",
        conn,
    )
    archive.to_csv(str(EXPORT_DIR / "archive.csv"), index=False)

    conn.close()
    log.info(f"Exported JSON + CSV to {EXPORT_DIR}")


def _write_json(path, data):
    """Write JSON with pretty printing."""
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"  Wrote {path.name}")
