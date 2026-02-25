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

    # ── 4. archive.csv — historical actuals + predictions ────────────
    # Include all auction dates (with any matching 7-day prediction),
    # plus forward forecasts whose target dates have no auction yet.
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
        UNION
        SELECT f2.target_date AS date,
               NULL AS actual_avg_price_inr_per_kg,
               f2.predicted_price AS predicted_avg_price_inr_per_kg,
               f2.forecast_date AS model_run_date,
               f2.horizon_days,
               f2.model_version
           FROM forecast_ledger f2
           WHERE f2.target_date NOT IN (SELECT date FROM auction_daily)
           ORDER BY date""",
        conn,
    )
    archive.to_csv(str(EXPORT_DIR / "archive.csv"), index=False)

    # ── 5. insights.json — market analysis for Insights page ─────
    _export_insights(conn)

    conn.close()
    log.info(f"Exported JSON + CSV to {EXPORT_DIR}")


def _export_insights(conn):
    """Build insights.json from full price history and external data."""
    import numpy as np

    # --- Full price history ---
    prices = pd.read_sql(
        "SELECT date, avg_price, max_price, sold_kg, arrived_kg FROM auction_daily ORDER BY date",
        conn,
    )
    if prices.empty:
        _write_json(EXPORT_DIR / "insights.json", {})
        return

    prices["date"] = pd.to_datetime(prices["date"])
    prices["year"] = prices["date"].dt.year
    prices["month"] = prices["date"].dt.month
    prices["month_name"] = prices["date"].dt.strftime("%b")

    # --- 1. Price history timeline (yearly summaries) ---
    yearly = prices.groupby("year").agg(
        avg_price=("avg_price", "mean"),
        max_price=("max_price", "max"),
        min_price=("avg_price", "min"),
        total_volume_kg=("sold_kg", "sum"),
        trading_days=("date", "count"),
    ).reset_index()
    yearly = yearly[yearly["trading_days"] >= 30]  # skip partial years

    yearly_timeline = []
    for _, row in yearly.iterrows():
        yearly_timeline.append({
            "year": int(row["year"]),
            "avg_price": round(float(row["avg_price"]), 0),
            "max_price": round(float(row["max_price"]), 0),
            "min_price": round(float(row["min_price"]), 0),
            "total_volume_mt": round(float(row["total_volume_kg"]) / 1000, 0),
        })

    # --- 2. Monthly seasonality (avg across all years) ---
    monthly_avg = prices.groupby("month")["avg_price"].mean()
    grand_mean = prices["avg_price"].mean()
    seasonality = []
    for m in range(1, 13):
        avg = monthly_avg.get(m, grand_mean)
        deviation = (avg - grand_mean) / grand_mean * 100
        seasonality.append({
            "month": m,
            "month_name": ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
            "avg_price": round(float(avg), 0),
            "deviation_pct": round(float(deviation), 1),
        })

    # Best / worst months
    sorted_months = sorted(seasonality, key=lambda x: x["deviation_pct"], reverse=True)
    best_months = [m["month_name"] for m in sorted_months[:3]]
    worst_months = [m["month_name"] for m in sorted_months[-3:]]

    # --- 3. Key price regimes ---
    regimes = [
        {
            "name": "Gradual Rise",
            "period": "Nov 2014 - Jul 2018",
            "start": "2014-11-01",
            "end": "2018-07-31",
            "driver": "Indian production growth, steady demand, cost floor rising",
            "direction": "up",
        },
        {
            "name": "Kerala Flood Spike",
            "period": "Aug 2018 - Dec 2020",
            "start": "2018-08-01",
            "end": "2020-12-31",
            "driver": "2018 Kerala floods (La Nina) destroyed crops, supply crunch",
            "direction": "up",
        },
        {
            "name": "Supply Glut Correction",
            "period": "Jan 2021 - Jun 2023",
            "start": "2021-01-01",
            "end": "2023-06-30",
            "driver": "New plantings from 2019-20 boom hit market, auction volumes surged",
            "direction": "down",
        },
        {
            "name": "Guatemala Crisis Bull Run",
            "period": "Jul 2023 - Present",
            "start": "2023-07-01",
            "end": "2099-12-31",
            "driver": "Guatemala thrips crisis collapsed 46-75% of global exports",
            "direction": "up",
        },
    ]
    for regime in regimes:
        mask = (prices["date"] >= regime["start"]) & (prices["date"] <= regime["end"])
        subset = prices.loc[mask]
        if not subset.empty:
            regime["start_price"] = round(float(subset["avg_price"].iloc[0]), 0)
            regime["end_price"] = round(float(subset["avg_price"].iloc[-1]), 0)
            regime["peak_price"] = round(float(subset["avg_price"].max()), 0)
            regime["trough_price"] = round(float(subset["avg_price"].min()), 0)
            change = (subset["avg_price"].iloc[-1] - subset["avg_price"].iloc[0]) / subset["avg_price"].iloc[0] * 100
            regime["change_pct"] = round(float(change), 1)
        # Clean internal fields
        del regime["start"]
        del regime["end"]

    # --- 4. Key drivers analysis ---
    # ENSO correlation
    enso = pd.read_sql("SELECT * FROM enso_monthly ORDER BY year, season", conn)
    enso_insight = None
    if not enso.empty:
        # Get latest ENSO reading
        latest_enso = enso.iloc[-1]
        phase = "El Nino" if latest_enso["anomaly"] > 0.5 else (
            "La Nina" if latest_enso["anomaly"] < -0.5 else "Neutral"
        )
        enso_insight = {
            "latest_anomaly": round(float(latest_enso["anomaly"]), 2),
            "latest_season": f"{int(latest_enso['year'])} {latest_enso['season']}",
            "phase": phase,
            "impact_note": "La Nina (excess rain) leads to crop damage and HIGHER prices with ~12 month lag. "
                           "El Nino (drier) tends to be manageable with LOWER prices.",
        }

    # Guatemala trade
    guatemala = pd.read_sql(
        "SELECT * FROM trade_guatemala ORDER BY period", conn
    )
    guatemala_insight = None
    if not guatemala.empty and len(guatemala) >= 12:
        gt = guatemala.copy()
        gt["qty_kg"] = pd.to_numeric(gt["qty_kg"], errors="coerce")
        gt_recent_12 = gt.tail(12)["qty_kg"].sum()
        gt_prev_12 = gt.iloc[-24:-12]["qty_kg"].sum() if len(gt) >= 24 else None
        yoy_change = None
        if gt_prev_12 and gt_prev_12 > 0:
            yoy_change = round((gt_recent_12 - gt_prev_12) / gt_prev_12 * 100, 1)
        guatemala_insight = {
            "recent_12m_qty_mt": round(float(gt_recent_12) / 1000, 0) if gt_recent_12 else None,
            "yoy_change_pct": yoy_change,
            "impact_note": "Guatemala supplies 60-70% of global cardamom exports. "
                           "The 2024-25 thrips crisis collapsed exports, driving Indian prices up.",
        }

    # Rainfall
    weather = pd.read_sql(
        "SELECT date, rain_mm FROM weather_idukki ORDER BY date DESC LIMIT 90", conn
    )
    rainfall_insight = None
    if not weather.empty:
        total_rain_90d = weather["rain_mm"].sum()
        rainfall_insight = {
            "last_90d_rain_mm": round(float(total_rain_90d), 0),
            "impact_note": "Idukki receives 2,500-3,500mm annually. "
                           "Excess monsoon rain causes flooding and crop damage. "
                           "Deficit stress reduces yields.",
        }

    # USD/INR
    finance = pd.read_sql(
        "SELECT date, usdinr FROM finance_daily ORDER BY date DESC LIMIT 30", conn
    )
    usdinr_insight = None
    if not finance.empty:
        latest_fx = finance.iloc[0]["usdinr"]
        oldest_fx = finance.iloc[-1]["usdinr"]
        fx_change = round((latest_fx - oldest_fx) / oldest_fx * 100, 2) if oldest_fx else 0
        usdinr_insight = {
            "current_rate": round(float(latest_fx), 2),
            "monthly_change_pct": fx_change,
            "impact_note": "Weaker rupee makes Indian cardamom cheaper globally, "
                           "boosting export demand and supporting prices.",
        }

    drivers = {
        "enso": enso_insight,
        "guatemala_supply": guatemala_insight,
        "rainfall": rainfall_insight,
        "exchange_rate": usdinr_insight,
    }

    # --- 5. Planting cycle analysis (best/worst years to plant) ---
    # A farmer who plants in year Y harvests in Y+3 and sells for ~5 years (Y+3 to Y+7).
    # Best planting years = years where the 3-7 year forward avg price was highest.
    planting_analysis = []
    for _, row in yearly.iterrows():
        y = int(row["year"])
        # Forward 3-7 year window (harvest period)
        harvest_years = yearly[(yearly["year"] >= y + 3) & (yearly["year"] <= y + 7)]
        if len(harvest_years) >= 2:
            fwd_avg = harvest_years["avg_price"].mean()
            cost_at_planting = 250 + (y - 2014) * 30  # rough cost escalation
            roi = (fwd_avg - cost_at_planting) / cost_at_planting * 100
            planting_analysis.append({
                "planting_year": y,
                "harvest_window": f"{y+3}-{y+7}",
                "avg_harvest_price": round(float(fwd_avg), 0),
                "estimated_roi_pct": round(float(roi), 0),
            })

    sorted_planting = sorted(planting_analysis, key=lambda x: x["avg_harvest_price"], reverse=True)

    # --- 6. Cost floor analysis ---
    # MGNREGA base wages and typical farm labour multiplier
    base_year = 2014
    base_wage = 212  # Kerala MGNREGA 2014
    wage_growth = 0.052  # 5.2% CAGR
    current_year = prices["date"].max().year
    labour_multiplier = 1.7  # farm labour premium over MGNREGA
    labour_share = 0.62  # labour = 62% of total cost

    cost_floor_history = []
    for y in range(base_year, current_year + 2):
        mgnrega = base_wage * (1 + wage_growth) ** (y - base_year)
        farm_wage = mgnrega * labour_multiplier
        total_cost = farm_wage / labour_share  # ₹/day implies cost/kg via yield conversion
        # Approximate cost/kg: ~₹350 base in 2014, scaling with wage
        cost_per_kg = 350 * (1 + wage_growth) ** (y - base_year)
        cost_floor_history.append({
            "year": y,
            "mgnrega_wage": round(float(mgnrega), 0),
            "estimated_cost_per_kg": round(float(cost_per_kg), 0),
        })

    current_cost = cost_floor_history[-2]["estimated_cost_per_kg"]  # last full year
    current_price = float(prices["avg_price"].iloc[-1])
    price_to_cost = round(current_price / current_cost, 1) if current_cost else None

    # --- 7. Farmer-relevant Q&A ---
    farmer_qa = [
        {
            "question": "When is the best time to sell my harvest?",
            "answer": f"Historically, {best_months[0]}-{best_months[1]} and {best_months[2]} "
                      f"fetch the highest prices ({sorted_months[0]['deviation_pct']:+.1f}% above annual average). "
                      f"Avoid selling in {worst_months[0]}-{worst_months[1]} when prices typically dip "
                      f"{sorted_months[-1]['deviation_pct']:.1f}% below average.",
            "category": "selling",
        },
        {
            "question": "Is it a good time to plant new cardamom?",
            "answer": f"Current prices are {price_to_cost}x the estimated cost floor — historically very "
                      f"profitable. However, high prices trigger planting booms. The 2024-25 planting wave "
                      f"will likely flood the market by 2027-28, potentially crashing prices. "
                      f"Consider planting when prices are depressed (1-2x cost) for better long-term returns.",
            "category": "planting",
        },
        {
            "question": "How does weather affect my crop and prices?",
            "answer": "La Nina (excess rain) causes flooding and crop damage in Idukki, leading to "
                      "higher prices 12 months later. El Nino (drier conditions) is generally manageable. "
                      "Monitor the NOAA ENSO index — transitions to La Nina are the biggest weather risk.",
            "category": "weather",
        },
        {
            "question": "Why are prices so high right now?",
            "answer": "The dominant driver is the Guatemala thrips crisis (2024-25), which collapsed "
                      "60-70% of global cardamom exports. Guatemala normally supplies the majority of "
                      "world cardamom. Until Guatemala recovers, Indian prices stay elevated.",
            "category": "market",
        },
        {
            "question": "When might prices fall significantly?",
            "answer": "Watch for three signals: (1) Guatemala export volumes recovering — "
                      "this would increase global supply, (2) Indian auction volumes rising sharply — "
                      "indicating new domestic supply, (3) Google Trends 'cardamom plantation' interest "
                      "declining — suggesting planting wave is maturing. Current regime risk is LOW.",
            "category": "risk",
        },
        {
            "question": "What is the cost floor for cardamom farming?",
            "answer": f"The estimated production cost is ~₹{int(current_cost)}/kg (2025), rising ~5% "
                      f"per year with labour inflation. Current prices at ₹{int(current_price)}/kg are "
                      f"{price_to_cost}x the cost floor. Prices rarely stay below 1.5x cost for long "
                      f"because farmers abandon cultivation, reducing future supply.",
            "category": "economics",
        },
    ]

    # --- Assemble insights ---
    insights = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "price_timeline": {
            "yearly_summary": yearly_timeline,
            "date_range": {
                "start": prices["date"].min().strftime("%Y-%m-%d"),
                "end": prices["date"].max().strftime("%Y-%m-%d"),
                "total_trading_days": len(prices),
            },
            "all_time_high": round(float(prices["avg_price"].max()), 0),
            "all_time_low": round(float(prices["avg_price"].min()), 0),
        },
        "seasonality": {
            "monthly": seasonality,
            "best_months": best_months,
            "worst_months": worst_months,
        },
        "regimes": regimes,
        "drivers": drivers,
        "planting_analysis": {
            "best_years": sorted_planting[:3] if sorted_planting else [],
            "worst_years": sorted_planting[-3:] if len(sorted_planting) >= 3 else [],
            "current_price_to_cost": price_to_cost,
            "cost_floor_history": cost_floor_history,
        },
        "farmer_qa": farmer_qa,
    }
    _write_json(EXPORT_DIR / "insights.json", insights)


def _write_json(path, data):
    """Write JSON with pretty printing."""
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"  Wrote {path.name}")
