# Prediction Performance Assessment

**Date:** 2026-08-05
**Scope:** 378 matured forecasts across 10 horizons, forecast run dates 2026-02-06 →
2026-08-05. (The settlement fix described below later raised this to 645 by scoring
forecasts the old exact-date rule had silently dropped.)

## Summary

The pipeline is well-engineered and the forecast ledger is honestly kept — forecasts are
written before outcomes exist, so a real out-of-sample assessment was possible. On that
evidence the forecasts carry **no measurable predictive value over "tomorrow's price is
today's price"**. Pooled directional accuracy is 50.0% and pooled predicted-vs-realized
return correlation is −0.064. The 90-day model is significantly *worse* than naive
(DM p < 0.001). Headline accuracy shown in the app is hardcoded and overstates measured
directional accuracy by roughly 38 points.

**Fixed in v2.3** (code merged, effective for the daily models after the next retrain):

| | Was | Now |
|---|---|---|
| Daily training horizon | `shift(-h)` stepped auction *sessions*: 8.82 days at h=7, 17.65 at h=14 | Settles on the first auction at or after `date + h days` — 7.10 and 14.10 |
| Forecast settlement | Exact `target_date` auction match only; 41.5% never scored | First auction at or after target date; scored forecasts 378 → 645 |
| Reported skill | Price-level MAPE alone, carried by the anchor | Naive baseline, skill, Theil's U, directional accuracy, beat-naive rate |
| `total_predictions` | Pinned at 200 by a `LIMIT 200` query | True count of the validation log |
| Per-horizon window | Advertised 30, actually ~16–23 | Genuine 30 per horizon |

**Investigated and rejected:** mean-reversion terms. The series is genuinely
mean-reverting, but that does not convert into forecast skill under walk-forward
validation — see below.

**Still open:** the app's hardcoded metric cards are not bound to `track_record.json`;
the 1–7 day product publishes point forecasts for what is close to a martingale;
`finance.py:92` raises `KeyError: 'date'` when the Yahoo fetch fails and it falls back
to CSV, which would take down a production run during a Yahoo outage.

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

## What would improve accuracy

Two structural bugs found while investigating this, both of which cost accuracy directly:

**1. The daily models are trained on the wrong horizon.** `make_return_target` applies
`df[price_col].shift(-horizon)` to a frame indexed by *auction sessions*, but
`predict_all` labels the result `today + timedelta(days=h)` — *calendar* days. At 5.14
auctions/week these diverge: `shift(-7)` averages **8.82 calendar days**, and `shift(-14)`
averages **17.65**. The models are trained to answer a question ~26% further out than the
one they are scored on. Weekly (28d) and monthly (90d) frames resample on calendar
boundaries and are unaffected.

**2. 41.5% of matured forecasts are never validated.** `validate.py` only fires on an
exact `target_date` match against an auction day, so forecasts targeting non-auction days
are silently dropped — 55% of 1-day forecasts, ~50% at 2–6 days. The published track
record is therefore a non-random subsample, not the full record.

**3. The series is mean-reverting, but that does not convert into forecast skill.**
Lag-1 return autocorrelation is **-0.129** (~7 SE, n=3150) and variance ratios run
**0.69–0.87** across q=2…60 — persistent multi-scale mean reversion.

An initial single-window test (coefficients fit pre-2026, scored on the Feb–Aug 2026
forecasts) suggested a one-feature rule beat the naive baseline at all 10 horizons.
**It does not survive proper validation.** Under walk-forward CV across the full
2014–2026 history and all regimes (`backtest_mean_reversion.py`), no configuration beats
the random walk at any horizon — the earlier window was a single regime with n=25–35 at
the long horizons.

Skill (`1 - MAE/MAE_naive`; >0 beats the random walk):

| horizon | T1 only (current) | T1 + reversion | reversion only, ridge |
|---:|---:|---:|---:|
| 1d  | −0.169 | −0.142 | **−0.002** |
| 7d  | −0.248 | −0.275 | **−0.000** |
| 14d | −0.483 | −0.441 | **−0.010** |
| 28d | −0.252 | −0.329 | **−0.021** |
| 90d | −0.589 | −0.588 | −0.132 (−0.096 Bayesian) |

The ordering is monotone and consistent: **fewer features and less model capacity move
you closer to the naive baseline**. Adding reversion terms to the existing feature set
barely helps and at 7d and 28d actively hurts. The problem is not a missing feature — it
is capacity far exceeding the available signal.

The actionable finding is therefore simplification rather than augmentation. A plain
ridge on reversion terms matches the naive baseline while the current stack is 17–59%
worse; at 14d that is MAPE 0.069 vs 0.093, and at 90d 0.208 vs 0.272.

**A contributing design flaw:** every T1 feature is computed on `price.shift(1)`
(`features.py:43-86`), including the mean-reversion ones. The anchor and the target are
both measured from *today's* price, so the models are asked to predict a move starting
from a level they cannot observe — today's deviation from its own moving average is
never available to them. Today's auction has closed by the 6 PM IST run and `predict_all`
already anchors on it, so exposing it is causal. Doing so helps at 1d and 14d but not at
7d; it is a real flaw, not the dominant one.

**On the composition-artifact question.** A natural worry is that reversion in a
volume-weighted average of heterogeneous lots is an artifact of daily composition drift
rather than economics. Rebuilding the index with auctioneer fixed effects (21
auctioneers, effects spanning 6.1%) makes reversion *stronger*, not weaker — lag-1
−0.149 vs −0.110, VR(10) 0.70 vs 0.72 — which is the opposite of what an artifact story
predicts. That rules out the **auctioneer** channel. It does **not** rule out grade-mix
drift *within* auctions (8mm AGEB vs 7mm AGB vs 6mm AGS), which is the more likely
channel and which the available data cannot test: the Spices Board XLS carries no
grade-level detail. Settling it needs a matched-grade or hedonic index built from
per-lot data we do not currently collect.

**Realistic expectations by horizon.** Mean absolute moves since Feb 2026 are 1.84%
(1 session), 4.92% (7), 9.22% (28), 23.71% (90). A perfect random walk scores about these
numbers, which is the bar any model must clear. At 1–7 days the series is close enough to
a martingale that the honest move may be to publish the last price with a calibrated
interval rather than a point forecast implying skill that is not there.

Suggested order of work: fix the horizon mismatch and the validation gap; add the naive
baseline to walk-forward CV and to `track_record.json` and report skill scores rather than
raw MAPE; bind the UI metrics to real data; then test mean-reversion terms at 28d/90d
against the naive baseline before adding any further model complexity.

## Reconciliation with the forecasting literature

A deep-research review of the field (prompt in `Deep_Research_Prompt_Forecasting_SOTA.md`)
was run against these diagnostics. Recording where it agrees, where measurement overrides
it, and where it identifies something we cannot yet test.

**Confirmed by measurement.** Price-level MAPE is an anchor-flattered illusion and should
be replaced by loss differentials against a baseline (implemented in v2.3). High-capacity
ML in a low signal-to-noise series is a bias-variance failure — our skill table shows a
monotone improvement as capacity falls. Walk-forward permutation top-k selection is itself
a source of selection bias. Slow-moving macro predictors (USD/INR, gold, Nifty) do not
belong in short-horizon models. The 1–7 day horizon is plausibly a near-martingale where
no method should be expected to add value — our walk-forward results agree.

**Where measurement overrides the review.** The review ranks a 20-day moving-average
linear rule first and expects it to "reliably beat the random walk" under Clark-West. It
does not. Measured over the full history it *matches* the naive baseline (skill −0.000 to
−0.021) and never beats it. The Schwartz-Smith short-term/long-term decomposition is a
useful explanation for why such a rule dominates the GBM stack, but it does not deliver
skill over persistence in this series.

**Where we disagree on method.** The review recommends forward-filling non-auction days
onto a continuous calendar index so that targets align. Doing so would make a Sunday
target equal to Friday's already-known price — a partially-known target that inflates
apparent accuracy, severely at the 1-day horizon. v2.3 instead settles against the first
*actual* auction at or after the target date, which always scores against a genuinely new
observation. We consider this the more honest construction and have kept it.

**Where the review identifies something we cannot currently test.** It attributes the
−0.129 lag-1 autocorrelation to lot-quality composition drift. Our two-way fixed-effects
test rules out the *auctioneer* channel — controlling for it makes reversion stronger, not
weaker. But the review points at grade-mix drift *within* auctions (8mm AGEB / 7mm AGB /
6mm AGS), which our test does not touch and which is the more plausible channel. The
Spices Board XLS carries no grade-level detail (only date, auctioneer, lots, quantities,
max and average price), so the recommended matched-grade and hedonic index tests are not
runnable on data we collect today. **Settling this requires a collection change, not an
analysis change** — and it determines whether short-horizon forecasting should exist at
all.

**Most valuable unimplemented recommendation.** Adaptive Conformal Inference for
prediction intervals. It is the honest way to ship a 1–7 day product that has no
point-forecast skill: distribution-free coverage that widens automatically when a regime
breaks, rather than a point number implying precision the model does not have.

**Worth noting on bias correction.** Mincer-Zarnowitz recalibration is well-motivated for
the 90-day model given its +0.565 correlation alongside a large one-sided bias. But it
needs matured outcomes, and at 90 days only ~4 arrive per year — a causal rolling
correction tested here never accumulated enough history to activate. Expect it to
calibrate slowly.

## Bottom line

The measurement scaffolding here is better than the models it measures, and that is what
made this assessment possible. The models themselves add nothing over persistence, and
adding features has not been the answer — every configuration tested moved *closer* to
the naive baseline as capacity fell, never past it.

The remaining work is therefore not a better model. In priority order:

1. **Bind the displayed metrics to `track_record.json`.** The app currently advertises 88%
   directional accuracy against a measured 50.0%. This is farmer-facing and is the one
   item with a real-world cost attached.
2. **Stop publishing long-horizon point forecasts that lose to persistence.** At 90 days
   the model is significantly worse than naive and carries a 100% one-sided error. Either
   default to the naive baseline or withhold the horizon until something beats it.
3. **Replace point forecasts with calibrated intervals at 1–7 days**, via Adaptive
   Conformal Inference. The series is close to a martingale there; an interval is an
   honest product, a point forecast is not.
4. **Collect grade-level auction data.** Whether short-horizon reversion is real or a
   grade-mix artifact cannot be settled with what is stored today, and the answer decides
   whether the short-horizon product has any basis at all.

Every claim in this document is reproducible from the repository:
`backtest_mean_reversion.py` for the skill tables, and the forecast ledger reconstructed
from the git history of `cardamom_webapp/data/archive.csv` for the live track record.
