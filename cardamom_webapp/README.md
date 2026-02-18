# Cardamom Price – Web Dashboard (No Backend)

This is a single-page web app for **tracking predictions vs. actual average Kerala auction prices** for cardamom, and scoring the deviation.

## How it works
- Open `index.html` in any browser (Chrome/Edge/Firefox).
- Click **Load archive.csv** to load your history of predictions vs actuals.
- Or upload **new predictions** and/or **actuals** CSVs to merge into the current archive.
- The page computes **daily deviation**, **30-day MAPE, RMSE, MAE**, and **hit-rate (±5%)**.
- Click **Download updated archive.csv** to save the merged data.

## CSV formats
- **Archive file:** `archive.csv` with columns:
  - `date` (YYYY-MM-DD)
  - `actual_avg_price_inr_per_kg`
  - `predicted_avg_price_inr_per_kg`
  - `model_run_date` (optional)
  - `horizon_days` (optional; e.g., 1 for T+1)
  - `model_version` (optional tag)

- **Predictions file:** at minimum: `date`, `predicted_avg_price_inr_per_kg` (+ optional metadata).
- **Actuals file:** at minimum: `date`, `actual_avg_price_inr_per_kg`.

> See `data/archive_template.csv` for a starter.

## Scoring
- Daily % error = |pred − actual| / actual
- Deviation score (0–100) = max(0, 100 × (1 − % error))
- Dashboard shows 30-day MAPE, RMSE, MAE, and hit-rate within ±5%.

## Notes
- Everything runs locally in your browser. **No data is uploaded anywhere.**
- The app uses two client-side libraries via CDN: **Papa Parse** (CSV parsing) and **Chart.js** (charts).