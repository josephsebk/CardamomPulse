# Prediction Performance Assessment

**Date:** 2026-08-05
**Scope:** 378 matured forecasts across 10 horizons, forecast run dates 2026-02-06 → 2026-08-05.

## How this was measured

The live forecast ledger (`data/cardamom.db`) is gitignored, so the evaluation set was
reconstructed from the 50 daily snapshots of `cardamom_webapp/data/archive.csv` in git
history. Each snapshot records the forecasts that were still pending on that date
(`target_date`, `horizon_days`, `predicted_price`, `model_run_date`), so the union across
snapshots recovers the forecast ledger for every horizon — not just the 7-day slice the
current `archive.csv` retains, and not subject to the 200-row cap in `track_record.json`.

Those forecasts were then joined to realized volume-weighted auction prices. Per-horizon
sample sizes and error levels reproduce the pipeline's own `track_record.json` numbers,
which confirms the reconstruction.

**Benchmark.** Models forecast a log return and reconstruct price as
`anchor × exp(return)`, where `anchor` is the last known auction price
(`pipeline/models.py:_last_row`). So the natural null hypothesis is the random walk —
the same anchor with `return = 0`, i.e. "assume no change". This is an exact
apples-to-apples comparison: it isolates the model's only contribution, the return.

## Headline result

**The models do not demonstrate skill over a random walk at any horizon.**

| Horizon | n | MAPE model | MAPE naive | Skill vs naive | Theil U | Direction |
|--------:|--:|-----------:|-----------:|---------------:|--------:|----------:|
| 1d  | 29  | 2.08%  | 1.86%  | −0.121 | 1.108 | 55.2% |
| 2d  | 32  | 2.77%  | 2.77%  | +0.006 | 1.003 | 53.1% |
| 3d  | 31  | 3.12%  | 3.01%  | −0.042 | 1.012 | 38.7% |
| 4d  | 31  | 3.72%  | 3.30%  | −0.134 | 1.125 | 61.3% |
| 5d  | 31  | 4.05%  | 3.65%  | −0.101 | 1.048 | 67.7% |
| 6d  | 31  | 4.05%  | 3.90%  | −0.026 | 0.950 | 51.6% |
| 7d  | 100 | 5.21%  | 4.59%  | −0.139 | 1.165 | 47.0% |
| 14d | 35  | 7.26%  | 7.24%  | +0.034 | 0.999 | 57.1% |
| 28d | 33  | 9.29%  | 8.58%  | −0.073 | 1.062 | 42.4% |
| 90d | 25  | 18.86% | 14.74% | −0.287 | 1.269 | 28.0% |

Skill = `1 − MAE_model / MAE_naive` (positive means the model wins). Theil U below 1
means the model wins. Direction is measured against the anchor; 50% is a coin flip.

- The model beats persistence at **2 of 10 horizons** (2d, 14d), by margins
  (+0.006, +0.034) far too small to be meaningful.
- The result is robust to the anchor convention. Re-running the baseline with a
  strictly-lagged anchor (last price *before* the run date, which handicaps the
  baseline) leaves the same 2 of 10.
- Diebold–Mariano tests (HLN small-sample correction) return p > 0.11 at every horizon
  except 90d — short and medium horizons are statistically **indistinguishable** from
  doing nothing.
- **90-day is significantly worse than naive** (DM p < 0.001), with 28.7% higher MAE
  and 28% directional accuracy.
- Pooled directional accuracy across all 378 forecasts is **50.0%**.

## Why the published MAPE looks better than the model is

The reported error is computed on price *levels*, but the model only forecasts the
*return*; the level is carried by the anchor. At 7 days:

- mean absolute realized move: 4.65%
- mean absolute return the model calls: 2.59%
- published price-level MAPE: 5.21%

Most of the apparent accuracy is the last known auction price, not the model. Measured
in return space — the only part the model supplies — pooled correlation between
predicted and realized return is **−0.064**, and at 9 of 10 horizons the return forecast
scores worse than simply predicting zero.

A "2% MAPE at 1 day" therefore reflects the fact that cardamom prices rarely move much
in a day, not forecasting skill. No naive baseline is computed anywhere in the project,
so this gap has not been visible. (The "baseline" in `backtest_features.py` is a
baseline *feature set*, not a baseline *forecast*.)

## Systematic bias at long horizons

At 28d and 90d, **100% of forecasts landed below the actual price** — every single one.
Over the evaluation window the market rallied from ₹2,451 to ₹3,059/kg (+24.8%), and the
long-horizon models missed all of it, with mean signed errors of −₹277 and −₹566/kg.

The 90-day case is the most interesting: predicted and realized returns correlate at
**+0.565**, so the model does carry relative information, but a large constant negative
bias pushes every forecast below the outcome, which is what destroys both its directional
accuracy (28%) and its error. That is a calibration failure rather than an absence of
signal, and it is the most promising thing here to fix.

## Reported metrics that do not match measured performance

The Model Performance page in `cardamom_webapp/index.html` displays hardcoded static
values, not the contents of `track_record.json`:

| Displayed | Source | Measured |
|---|---|---|
| "88%" Directional Accuracy (line 916) | hardcoded | **50.0%** |
| "9.2%" MAPE (line 929) | hardcoded | 5.7% pooled |
| "High" Confidence Score (line 941) | hardcoded | — |
| "99.8%" Data Completeness, "Low" Model Drift | hardcoded | — |
| "High Accuracy" KPI (line 646) | hardcoded, `id` never assigned | — |
| model "v2.4" (line 906) | hardcoded | pipeline emits `v2.2` |

`track_record.json` *is* fetched and does drive the validations table, but the three
headline metric cards carry no `id` and no JS writes to them, so they cannot update.

Two further issues in `pipeline/export.py`:

- `total_predictions` reports `len()` of a query with `LIMIT 200`, so it is pinned at
  200 once the log passes 200 rows rather than being a true count.
- Per-horizon metrics take `subset.head(30)` intending a 30-observation window, but the
  200-row global cap is split across 10 horizons, so no horizon ever reaches 30
  (actual n: 16–23). The window silently shrinks as horizons are added.

## What is sound

Worth stating plainly, because the measurement scaffolding is better than the models:

- **Validation is genuinely out-of-sample.** Forecasts are written to an immutable
  `forecast_ledger` before outcomes exist, and `validate.py` compares them to the actual
  on the target date. No lookahead, no retrospective fitting. This is an honest track
  record — the reason a real assessment was possible at all.
- **Actuals are real.** `avg_price` is a volume-weighted mean across same-day
  auctioneers, not interpolated or filled.
- **Training methodology is reasonable** — walk-forward CV with purging, out-of-sample
  permutation feature selection, log-return targets, and a sanity guard that aborts the
  run on implausible short-horizon deviations.
- **The 90-day 80% prediction intervals are honest, even conservative**: published
  half-width ±29.9% against a realized 80th-percentile error of 23.9%. They are, however,
  near-constant (±29.7–30.0%), so they do not adapt to conditions.

## Bottom line

The pipeline is well-engineered and the track record is honestly kept, but on this
evidence the forecasts carry no measurable predictive value over "tomorrow's price is
today's price". The 90-day model is actively worse than that, and is the one publishing
the widest numbers to farmers. The headline accuracy figures shown in the app are
hardcoded and overstate measured directional accuracy by roughly 38 points.

Before this is relied on for selling decisions, the naive baseline should become a
permanent part of the reported track record, and the displayed metrics should be bound
to `track_record.json`.
