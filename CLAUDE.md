# CardamomPulse

Cardamom price prediction platform for Indian small cardamom (Alleppey Green, Kerala auctions). Combines ML forecasting with a static web dashboard.

## Project Structure

```
CardamomPulse/
├── pipeline/                    # Python ML pipeline (Chat 1: Model & Data)
│   ├── run_daily.py             # Orchestrator: collect -> assemble -> validate -> predict -> export
│   ├── config.py                # Paths, URLs, model version (currently v1.0)
│   ├── db.py                    # SQLite schema + helpers (data/cardamom.db)
│   ├── assemble.py              # Joins raw tables into daily/weekly/monthly DataFrames
│   ├── features.py              # Feature engineering (T1-T6 feature tiers)
│   ├── models.py                # GBM/Ridge/BayesianRidge training, walk-forward CV, prediction
│   ├── validate.py              # Compares past predictions vs actual auction prices
│   ├── export.py                # Generates JSON/CSV for the webapp
│   └── collectors/              # Data source scrapers
│       ├── auction.py           # Kerala auction prices (primary data)
│       ├── weather.py           # Idukki + Guatemala weather (Open-Meteo)
│       ├── finance.py           # USD/INR, crude oil, gold, Nifty (Yahoo Finance)
│       ├── enso.py              # ENSO ONI index (NOAA)
│       └── trade.py             # Guatemala exports, Saudi imports, Google Trends, festivals
│
├── cardamom_webapp/             # Static web frontend (Chat 2: Frontend)
│   ├── index.html               # Single-page app (~2300 lines: HTML + CSS + JS)
│   └── data/                    # JSON/CSV consumed by the frontend
│       ├── dashboard.json       # Spot price, forecasts, regime
│       ├── price_history.json   # Last 90 days of auction prices
│       ├── track_record.json    # Model validation metrics by horizon
│       ├── insights.json        # Yearly summaries, seasonality, drivers, planting, Q&A
│       ├── archive.csv          # Full historical actuals + all predictions (immutable ledger)
│       └── archive_template.csv # CSV schema reference
│
├── data/
│   ├── cardamom.db              # SQLite database (all raw + processed data)
│   └── models/                  # Trained model files (.pkl)
│       ├── model_{1-7}d.pkl     # Daily horizon models (GBM)
│       ├── model_14d.pkl        # 2-week model
│       ├── model_28d_lgb.pkl    # Monthly model (LightGBM ensemble)
│       ├── model_28d_ridge.pkl  # Monthly model (Ridge component)
│       ├── model_90d.pkl        # 3-month model
│       ├── model_regime.pkl     # Bull/Bear/Neutral classifier
│       ├── scaler_28d.pkl       # StandardScaler for 28d features
│       ├── scaler_90d.pkl       # StandardScaler for 90d features
│       ├── medians_90d.pkl      # Median imputation values
│       └── medians_regime.pkl   # Median imputation for regime model
│
├── external_*.csv               # Raw external data files (repo root)
│   ├── external_idukki_weather.csv
│   ├── external_guatemala_weather.csv
│   ├── external_usdinr.csv
│   ├── external_crude_oil.csv
│   ├── external_gold.csv
│   ├── external_nifty50.csv
│   ├── external_enso_oni.csv
│   ├── external_comtrade_annual.csv
│   ├── external_faostat_*.csv
│   ├── external_guatemala_monthly_exports.csv
│   ├── external_saudi_monthly_imports.csv
│   ├── external_uae_monthly_imports.csv
│   ├── external_google_trends_cardamom.csv
│   └── external_festival_calendar.csv
│
└── requirements.txt             # Python deps: pandas, scikit-learn, requests, etc.
```

## Architecture Overview

### Pipeline Flow (run_daily.py)
1. **COLLECT** - Scrape auction results, weather, finance, ENSO, trade data
2. **ASSEMBLE** - Join into daily/weekly/monthly DataFrames via SQLite
3. **VALIDATE** - Compare yesterday's predictions against today's actual price
4. **PREDICT** - Run all models (1d-90d + regime) on latest features
5. **ARCHIVE** - Store predictions in immutable forecast_ledger table
6. **EXPORT** - Generate static JSON/CSV files for the web dashboard

### Pipeline Commands
```bash
python -m pipeline.run_daily                # Normal daily run
python -m pipeline.run_daily --seed         # First-time: seed DB from CSVs + train
python -m pipeline.run_daily --retrain      # Force full model retrain
python -m pipeline.run_daily --export-only  # Just regenerate JSON exports
```

### Database (SQLite)
Key tables:
- `auction_daily` — date, lots, arrived_kg, sold_kg, avg_price, max_price
- `forecast_ledger` — forecast_date, target_date, horizon_days, predicted_price, bounds
- `validation_log` — date, horizon_days, predicted_price, actual_price, abs_error, pct_error
- `regime_ledger` — forecast_date, bear_probability, regime_label
- `weather_idukki`, `weather_guatemala`, `finance_daily`, `enso_monthly`
- `trade_guatemala`, `trade_saudi`, `google_trends`, `festival_calendar`

### Models
- **1d-7d**: GradientBoostingRegressor on daily features (T1 + T2)
- **14d**: GradientBoostingRegressor on T1 + full T2
- **28d**: Ridge + LightGBM ensemble on T1-T5 features, with StandardScaler
- **90d**: GradientBoostingRegressor on T1 subset + T2 monthly + T3/T5/T6
- **Regime**: GradientBoostingClassifier (bull/bear/neutral)
- Walk-forward CV with purge gap to prevent leakage
- Model version tracked as `v1.0` in config.py

### Frontend (index.html)
Single-page static app with 4 tabs:
- **Dashboard** — KPI cards (spot, predicted, range, regime), forecast chart, intelligence feed, 90-day history
- **Intelligence** — Model performance metrics, predicted vs actual chart, forecast ledger table
- **Insights** — Price history timeline, market regimes, seasonality, drivers, planting analysis, farmer Q&A
- **Plans** — Pricing tiers (Free / Standard / Report)

Uses Chart.js for charts, PapaParse for CSV parsing. All data loaded from `cardamom_webapp/data/` JSON/CSV files.

## Data Contract (Pipeline -> Frontend)

The pipeline exports 5 files to `cardamom_webapp/data/`:

### archive.csv
```
date, actual_avg_price_inr_per_kg, predicted_avg_price_inr_per_kg, model_run_date, horizon_days, model_version
```
- Rows with actual price only = historical auction data
- Rows with predicted price only = future forecasts (pending validation)
- Rows with both = validated predictions
- This is the primary data source for the frontend's ledger table and charts

### dashboard.json
```json
{
  "spot_price": { "date", "avg_price", "max_price", "daily_change", "daily_change_pct", "volume_kg" },
  "forecasts": [{ "horizon_days", "target_date", "predicted_price", "lower_bound", "upper_bound" }],
  "regime": { "bear_probability", "label", "forecast_date" }
}
```

### track_record.json
```json
{
  "total_predictions": N,
  "metrics_by_horizon": { "7": { "n", "mape", "mae", "rmse", "hit_5pct" }, ... },
  "recent_validations": [{ "date", "horizon_days", "predicted_price", "actual_price", "abs_error", "pct_error" }]
}
```

### price_history.json
Array of last 90 auction days: `{ date, avg_price, max_price, sold_kg }`

### insights.json
Yearly summaries, seasonality, regimes, drivers (ENSO, Guatemala, rainfall, FX), planting analysis, farmer Q&A.

## Key Domain Context

- **Commodity**: Small cardamom (Alleppey Green), traded at Kerala auctions
- **Currency**: INR per kg
- **Current regime** (as of Mar 2026): Bullish — driven by Guatemala thrips crisis (60-70% of global exports collapsed)
- **Price range**: Recent prices ~2,200-2,500/kg; ATH was during 2018-20 Kerala floods
- **Seasonality paradox**: Prices peak during harvest (Aug-Sep, Dec-Jan) due to festival demand, not during lean months
- **Cost floor**: ~550/kg (2025), rising ~5%/yr with Kerala labour inflation
- **Model accuracy** (validated): 0.04% to 2% error on short horizons (1d-7d), larger on shocks

## Session-Specific Notes

### For ML Pipeline sessions (Chat 1)
- Models are in `pipeline/models.py`, features in `pipeline/features.py`
- Feature tiers: T1 (price lags/MAs), T2 (weather), T3 (trade/supply), T4 (finance), T5 (ENSO), T6 (trends/festivals)
- DB at `data/cardamom.db`, models at `data/models/*.pkl`
- Validation happens in `pipeline/validate.py` — checks forecast_ledger against auction_daily
- March 4, 2026 saw a sharp -5.5% drop (2,420 -> 2,288) that models missed

### For Frontend sessions (Chat 2)
- Everything is in `cardamom_webapp/index.html` (single file: CSS + HTML + JS)
- Charts use Chart.js 4.4.1 with date-fns adapter
- Data loaded via fetch from `cardamom_webapp/data/*.json` and PapaParse for CSV
- Intelligence page metric cards (88% accuracy, 9.2% MAPE) are currently hardcoded — should be driven from track_record.json
- Paywall overlay exists but is cosmetic (no auth backend)
- Mobile responsive breakpoints at 900px and 600px

### Known Issues / TODOs
- Intelligence page metrics are hardcoded, not computed from actual validation data
- Dashboard predicted price card doesn't show last validated accuracy
- No multi-horizon toggle on forecast chart (only shows one prediction line)
- No alert system for large price moves (>3% daily)
- Plans page "Report" tier missing /month or /one-time label
