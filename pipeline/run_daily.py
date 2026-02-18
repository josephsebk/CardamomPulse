#!/usr/bin/env python3
"""
CardamomPulse Daily Pipeline Orchestrator.

Run sequence (designed for 6:00 PM IST cron):
  1. COLLECT  — Fetch auction results + weather + financial data
  2. STORE    — Upsert into SQLite database
  3. VALIDATE — Compare yesterday's predictions against today's actual
  4. PREDICT  — Run all models on latest features
  5. ARCHIVE  — Store predictions in immutable forecast ledger
  6. EXPORT   — Generate JSON files for web dashboard
  7. (DEPLOY) — Handled externally (Vercel/Cloudflare push)

Usage:
  python -m pipeline.run_daily                   # Normal daily run
  python -m pipeline.run_daily --seed            # First-time: seed DB from CSV files
  python -m pipeline.run_daily --retrain         # Force full model retrain
  python -m pipeline.run_daily --export-only     # Just regenerate JSON exports
"""

import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


def step_init():
    """Initialize database schema."""
    from pipeline.db import init_db
    from pipeline.config import ensure_dirs
    ensure_dirs()
    init_db()
    log.info("Database initialized")


def step_collect():
    """Collect all data sources."""
    from pipeline.collectors.auction import collect_auction
    from pipeline.collectors.weather import collect_weather_idukki, collect_weather_guatemala
    from pipeline.collectors.finance import collect_finance
    from pipeline.collectors.enso import collect_enso
    from pipeline.collectors.trade import (
        collect_guatemala_exports, collect_saudi_imports,
        collect_google_trends, collect_festival_calendar,
    )

    log.info("═══ STEP 1: COLLECT ═══")
    collect_auction()
    collect_weather_idukki()
    collect_weather_guatemala()
    collect_finance()
    collect_enso()
    collect_guatemala_exports()
    collect_saudi_imports()
    collect_google_trends()
    collect_festival_calendar()
    log.info("Collection complete")


def step_assemble_and_engineer():
    """Assemble data and build features."""
    from pipeline.assemble import build_daily_df, resample_weekly, resample_monthly
    from pipeline.features import build_daily_features, build_weekly_features, build_monthly_features

    log.info("═══ STEP 2: ASSEMBLE & ENGINEER ═══")
    daily_raw = build_daily_df()
    if daily_raw.empty:
        log.error("No daily data assembled — aborting")
        return None, None, None

    weekly_raw = resample_weekly(daily_raw)
    monthly_raw = resample_monthly(daily_raw)

    daily, _ = build_daily_features(daily_raw)
    weekly, _ = build_weekly_features(weekly_raw)
    monthly, _ = build_monthly_features(monthly_raw)

    log.info(f"Features built: daily={daily.shape}, weekly={weekly.shape}, monthly={monthly.shape}")
    return daily, weekly, monthly


def step_validate(today: str):
    """Validate past predictions against today's actual."""
    from pipeline.validate import validate_predictions
    log.info("═══ STEP 3: VALIDATE ═══")
    results = validate_predictions(today)
    if results:
        for r in results:
            log.info(f"  {r['horizon_days']}d: pred=₹{r['predicted']:.0f} "
                     f"actual=₹{r['actual']:.0f} err={r['pct_error']:.1%}")
    else:
        log.info("  No predictions to validate today")


def step_train(daily, weekly, monthly):
    """Train all models (full retrain)."""
    from pipeline.models import train_all
    log.info("═══ STEP 4a: TRAIN ═══")
    results = train_all(daily, weekly, monthly)
    for horizon, info in results.items():
        cv = info.get("cv", {})
        if cv:
            log.info(f"  {horizon}: MAPE={cv.get('mape', 0):.3f} "
                     f"MAE={cv.get('mae', 0):.1f} R²={cv.get('r2', 0):.3f}")
    return results


def step_predict(daily, weekly, monthly, models):
    """Generate predictions and store in ledger."""
    from pipeline.models import predict_all, load_models
    from pipeline.db import get_conn
    from pipeline.config import MODEL_VERSION

    log.info("═══ STEP 4b: PREDICT ═══")

    # Load models (use freshly trained if available, otherwise from disk)
    if not models:
        models = load_models()
        if not models:
            log.error("No trained models found — run with --retrain first")
            return None

    forecasts = predict_all(daily, weekly, monthly, models)

    # Store in forecast ledger (immutable)
    conn = get_conn()
    for pred in forecasts.get("predictions", []):
        conn.execute(
            """INSERT OR REPLACE INTO forecast_ledger
               (forecast_date, target_date, horizon_days, predicted_price,
                lower_bound, upper_bound, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                forecasts["forecast_date"],
                pred["target_date"],
                pred["horizon_days"],
                pred["predicted_price"],
                pred.get("lower_bound"),
                pred.get("upper_bound"),
                MODEL_VERSION,
            ),
        )

    # Store regime
    regime = forecasts.get("regime")
    if regime:
        conn.execute(
            """INSERT OR REPLACE INTO regime_ledger
               (forecast_date, bear_probability, regime_label, model_version)
               VALUES (?, ?, ?, ?)""",
            (
                forecasts["forecast_date"],
                regime["bear_probability"],
                regime["label"],
                MODEL_VERSION,
            ),
        )

    conn.commit()
    conn.close()
    log.info(f"Stored {len(forecasts.get('predictions', []))} predictions in ledger")
    return forecasts


def step_export():
    """Generate JSON/CSV exports for the web frontend."""
    from pipeline.export import export_json
    log.info("═══ STEP 5: EXPORT ═══")
    export_json()


def run_pipeline(seed: bool = False, retrain: bool = False, export_only: bool = False):
    """Run the full pipeline."""
    start = datetime.now()
    log.info(f"{'='*60}")
    log.info(f"CardamomPulse Pipeline — {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*60}")

    # Always init DB
    step_init()

    if export_only:
        step_export()
        log.info(f"Export-only run complete in {(datetime.now()-start).seconds}s")
        return

    # Collect data
    step_collect()

    # Assemble + features
    daily, weekly, monthly = step_assemble_and_engineer()
    if daily is None:
        return

    today = daily["date"].max()

    # Validate yesterday's predictions
    step_validate(today)

    # Train (always on first seed, otherwise on --retrain or weekly)
    models = None
    if seed or retrain:
        models = step_train(daily, weekly, monthly)
    else:
        from pipeline.models import load_models
        models = load_models()
        if not models:
            log.info("No saved models found — training from scratch")
            models = step_train(daily, weekly, monthly)

    # Predict
    step_predict(daily, weekly, monthly, models)

    # Export
    step_export()

    elapsed = (datetime.now() - start).seconds
    log.info(f"{'='*60}")
    log.info(f"Pipeline complete in {elapsed}s")
    log.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="CardamomPulse Daily Pipeline")
    parser.add_argument("--seed", action="store_true",
                        help="First-time run: seed DB from CSV files and train models")
    parser.add_argument("--retrain", action="store_true",
                        help="Force full model retrain")
    parser.add_argument("--export-only", action="store_true",
                        help="Only regenerate JSON exports")
    args = parser.parse_args()

    try:
        run_pipeline(seed=args.seed, retrain=args.retrain, export_only=args.export_only)
    except Exception:
        log.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
