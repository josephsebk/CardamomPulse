# CardamomPulse — Project Overview & Commodity Replication Guide

A daily, automated **price-forecasting and market-intelligence pipeline** for the
Indian small-cardamom auction market. It scrapes auction prices, fuses ~9 external
data feeds (weather, ENSO, trade flows, macro, search interest), engineers a tiered
feature set, trains a family of gradient-boosted and Bayesian models on **log-return
targets**, validates every forecast against the realised price, and exports static
JSON for a no-backend web dashboard.

This document is written with two audiences in mind:

1. **Maintainers** of CardamomPulse who need the full architecture, the modelling
   decisions and *why* they were made, and the validated track record.
2. **Anyone replicating this for another commodity (e.g. coffee).** Sections 8–10
   are a concrete porting guide: what transfers as-is, what must be re-sourced, and
   what to drop or add.

> **One-line thesis of the whole project:** on a *single, ~3,000-point, covariate-rich*
> daily price series, well-engineered **gradient-boosted trees on log-return targets
> with rigorous walk-forward validation** beat everything more exotic (LSTM, TFT,
> foundation models). The moat is not model sophistication — it is the **disciplined
> data pipeline, the causal feature engineering, and the public, immutable track
> record.**

---

## 1. Repository Map

```
CardamomPulse/
├── pipeline/                     # The production system (all live code)
│   ├── run_daily.py              # Orchestrator: COLLECT→STORE→VALIDATE→PREDICT→ARCHIVE→EXPORT
│   ├── config.py                 # Paths, URLs, CV windows, feature-selection k, model version
│   ├── db.py                     # SQLite schema + upsert/read helpers
│   ├── collectors/               # One module per data source
│   │   ├── auction.py            # Spices Board scrape + XLS history → microstructure
│   │   ├── weather.py            # Open-Meteo (Idukki + Guatemala)
│   │   ├── finance.py            # Yahoo Finance (USD/INR, crude, gold, Nifty)
│   │   ├── enso.py               # NOAA CPC Oceanic Niño Index
│   │   └── trade.py              # Guatemala exports, Saudi imports, Google Trends, festivals
│   ├── assemble.py               # Merge all sources → daily / weekly / monthly DataFrames
│   ├── features.py               # 6-tier feature engineering (T1–T6 + microstructure)
│   ├── models.py                 # Targets, walk-forward CV, feature selection, train, predict
│   ├── validate.py               # Compare matured forecasts vs actuals → validation_log
│   └── export.py                 # Build dashboard/track-record/insights JSON + archive CSV
├── cardamom_webapp/              # Static front-end (HTML + Chart.js) + exported data/
├── external_*.csv                # Seed history for each feed (bootstraps the DB)
├── Small Cardamom Auction Prices.xls   # 5,440 per-auctioneer rows, Nov 2014–Feb 2026
├── *.ipynb                       # Research notebooks (exploration that fed the pipeline)
├── CardamomPulse_PRD.md          # Product requirements + validated model benchmarks
├── Auction_Microstructure_Insights.md  # Deep-dive on cross-auction signals
├── crontab.txt                   # Daily / weekly-retrain / monthly schedule
└── requirements.txt              # pandas, numpy, scikit-learn, joblib, requests, ...
```

There is **no backend server**. The pipeline is a cron-driven batch job that writes
JSON; the web app is static files served from a CDN.

---

## 2. Data Flow (end to end)

```
                ┌─────────────── COLLECT (collectors/*) ───────────────┐
  Spices Board ─┤ scrape today's per-auctioneer rows + load XLS history │
  Open-Meteo  ──┤ Idukki & Guatemala daily weather                      │
  Yahoo Fin.  ──┤ USD/INR, Brent, Gold, Nifty                           │
  NOAA CPC    ──┤ ENSO ONI (monthly)                                    │
  Comtrade    ──┤ Guatemala exports, Saudi imports (monthly CSV)        │
  Google Trends┤ search interest (monthly CSV)                          │
  Static CSV  ──┤ festival / season calendar                           │
                └───────────────────────┬───────────────────────────────┘
                                         ▼
                        STORE → SQLite (db.py, 11 tables)
                                         ▼
                ASSEMBLE (assemble.py): merge_asof(backward) for laggy
                monthly feeds, ffill macro gaps → daily / weekly / monthly
                                         ▼
                FEATURE ENGINEERING (features.py): 6 tiers, all causal
                (everything .shift(1)+rolling so row t never sees price t)
                                         ▼
                VALIDATE (validate.py): any forecast whose target_date == today
                gets scored vs the realised auction price → validation_log
                                         ▼
                TRAIN (weekly) / PREDICT (daily) (models.py):
                log-return targets, walk-forward CV, permutation feature
                selection, per-horizon model family
                                         ▼
                ARCHIVE → forecast_ledger (immutable) + regime_ledger
                                         ▼
                EXPORT (export.py) → dashboard.json, price_history.json,
                track_record.json, insights.json, archive.csv
                                         ▼
                Static web app (cardamom_webapp/) reads the JSON
```

---

## 3. The Database (`pipeline/db.py`)

SQLite, one file (`data/cardamom.db`), WAL mode. Key tables:

| Table | Grain | Purpose |
|---|---|---|
| `auction_daily` | daily | Price + volume + **microstructure** (n_auctions, price_disp_cv, price_range_pct, unsold_pct, avg_lot_kg) |
| `weather_idukki` / `weather_guatemala` | daily | Rain, temp, humidity at the two growing regions |
| `finance_daily` | daily | USD/INR, crude, gold, Nifty |
| `enso_monthly` | monthly | NOAA ONI anomaly by 3-month season |
| `trade_guatemala` / `trade_saudi` | monthly | Export/import qty + value (supply/demand proxies) |
| `google_trends` | monthly | Search interest by keyword |
| `festival_calendar` | daily | Binary flags for demand seasons |
| `forecast_ledger` | per forecast | **Immutable** record of every prediction (the trust moat) |
| `regime_ledger` | daily | Bear-probability history |
| `validation_log` | per matured forecast | predicted vs actual, abs/pct error |

Two design choices worth copying:

- **The forecast ledger is append-only and timestamped.** Every prediction ever made
  is retained with its `model_version`, so the public track record cannot be quietly
  rewritten. This is the product's defensibility, not the model accuracy.
- **Schema auto-migration.** `init_db()` `ALTER TABLE`s in new columns when an older DB
  is found (this is how microstructure columns were added without a wipe).

---

## 4. Data Collection (`pipeline/collectors/`)

Each collector follows the same robust pattern: **CSV seed history + live API top-up + idempotent upsert.**

- **`auction.py`** — the heart. Loads the 5,440-row XLS history, scrapes today's ticker
  from indianspices.com via regex, concatenates, de-dupes on (date, auctioneer), then
  `aggregate_daily()` collapses per-auctioneer rows to a **volume-weighted daily price**
  *and* computes the cross-auction microstructure signals that plain averaging destroys
  (dispersion CV, unsold share, auction count, lot size).
- **`weather.py`** — Open-Meteo archive API (free, no key). Determines the last stored
  date and only fetches the gap forward. Two locations: Idukki (Kerala, India) and
  Cobán (Guatemala).
- **`finance.py`** — Yahoo Finance v8 chart API directly via `requests` (no yfinance
  dependency). Forward-fills weekend/holiday gaps.
- **`enso.py`** — NOAA CPC ASCII file; maps 3-month season codes (DJF…NDJ) to a
  mid-month anchor, forward-fills to daily.
- **`trade.py`** — Comtrade-derived monthly CSVs (Guatemala exports, Saudi imports),
  Google Trends CSV, festival calendar CSV.

**Resilience patterns to copy:** every fetch is wrapped in try/except and degrades to
the CSV seed; a failed scrape never crashes the run; the latest auction XLS is the
manual fallback if the scraper breaks.

---

## 5. Feature Engineering (`pipeline/features.py`) — 6 Tiers

Everything is **causal**: features use `price.shift(1).rolling(...)` so the row for day
*t* is built only from information available *before* day *t*. This is the single most
important correctness property in the whole repo.

| Tier | Count | Used by | Examples | Intuition |
|---|---|---|---|---|
| **T1 Technical/Momentum** | ~33 | all | lags(1–30d), MA(7–90d), std, min/max, RSI(14), Bollinger position, volume ratio, sell-through, max-avg spread | autocorrelation + mean reversion |
| **T1b Microstructure** | 7 | 1–28d | `disp_cv_ma14`, `unsold_ma14`, `n_auctions_ma14`, `lot_size_rel` | cross-auction disagreement leads short-term moves |
| **T2 Calendar/Seasonal** | 13 d / 6 m | all | month/week sin-cos, festival flags, lean/strong season | demand seasonality |
| **T3 Weather/ENSO** | 14 | 3–8wk, 90d, regime | cumulative rainfall (28–182d) at both origins, ENSO + 6m/12m lags, ENSO phase | supply shocks with long lags |
| **T4 Macro** | 6 | 3–8wk | USD/INR level + %Δ, crude/gold/Nifty 28d %Δ (changes, not levels) | FX → export competitiveness |
| **T5 Trade/Supply** | 5 | 3–8wk, 90d, regime | Guatemala 3m export qty + YoY, Saudi import qty + YoY, unit price | global supply/demand balance |
| **T6 Structural/Cycle** | 6 | 90d, regime | cost floor (MGNREGA wage model), **months-since-trough cobweb cycle**, Google Trends 12m MA | multi-year planting cycle |

**Hard-won feature lessons (these are the real IP):**

1. **Use changes, not levels, for macro features.** Crude/gold/Nifty enter as 28-day %
   change, never raw level — levels are non-stationary and the tree memorises the era.
2. **Cobweb-cycle features must be built causally.** The original centered-window trough
   detector leaked future prices (a trough is only "known" after later prices confirm
   it). The causal rebuild — trailing-only smoothing, troughs confirmed only after `w`
   further periods — was worth a real accuracy jump on long horizons (see §7).
3. **Microstructure is a daily/weekly signal only.** Monthly averaging destroys it; it
   actually *degraded* the regime model (AUC 0.778 → 0.712), so it is excluded from the
   monthly pool.
4. **The 14-day mean of cross-auction price dispersion (`disp_cv_ma14`) is a genuine
   leading indicator** — top-3 permutation importance at 7d and 14d, and it leans
   bullish (high disagreement precedes upward moves).

---

## 6. The Models (`pipeline/models.py`)

### 6.1 Target: log-returns, not price levels

```python
target = log(price[t+h] / price[t])         # train on this
published_price = price[t] * exp(prediction) # reconstruct for display
```

**Why this matters more than any model choice:** tree ensembles *cannot predict outside
the range of target values seen in training*. A model trained on price *levels* caps out
at the historical high — exactly when forecasts matter most (record bull runs). Returns
are roughly stationary across regimes, removing that ceiling. This single change cut
walk-forward MAPE by **30–51%** on the 1–28 day horizons. If you replicate nothing else,
replicate this.

### 6.2 Model family (one model per horizon, by data grain)

| Horizon | Grain | Rows | Model | Uncertainty |
|---|---|---|---|---|
| 1–7 day | daily | ~3,000 | `GradientBoostingRegressor` (shallow, depth 4) | MAPE-derived band |
| 14 day | daily | ~3,000 | `GradientBoostingRegressor` | MAPE-derived band |
| 28 day | weekly | ~550 | **Stacked** GBR + Ridge (50/50 average) | MAPE-derived band |
| 90 day | monthly | ~130 | `BayesianRidge` | model-native 80% interval (log-space → asymmetric in price) |
| Regime | monthly | ~130 | `GradientBoostingClassifier` ("price drop >10% within 6 months") | calibrated probability |

The model *shrinks* as data shrinks: deep-ish trees on 3,000 daily rows, a
tree+linear stack on 550 weekly rows, and a fully linear Bayesian model on 130 monthly
rows. **Match model capacity to sample size** — this is why the 90-day uses Bayesian
Ridge and not another GBM.

### 6.3 Walk-forward cross-validation (never random splits)

`walk_forward_cv()` uses an **expanding window with a purge gap** equal to the forecast
horizon, so the train set never overlaps the label window. Metrics are computed on
*reconstructed price levels* so MAPE/MAE stay comparable across model versions. The
regime classifier additionally tracks walk-forward AUC/Brier on pooled out-of-sample
probabilities. Config in `config.py:WF_CONFIG`.

### 6.4 Permutation feature selection (out-of-sample)

With 69 candidate features on 534 weekly rows, naive training **overfit to negative R²**.
The fix (`select_features`): rank candidates by **mean permutation importance on the
held-out window of each walk-forward fold**, keep top-k per horizon (15 for 1–7d, 10 for
14d, 5 for 90d, 20 for regime; the 28d keeps a hand-curated T1–T5 set that beat every
selected subset). Out-of-sample importance avoids the impurity-bias trap that favours
high-cardinality noise features. Selected lists are persisted with the model in
`meta.pkl` and reused at prediction.

### 6.5 Operational guardrails (learned from outages)

- **Stale-anchor guard.** A log-return forecast is `anchor × exp(return)`; anchoring on
  an old row silently shifts every forecast. `predict_all` always anchors on the latest
  *priced* row and ffill/median-imputes residual NaNs. `_sanity_check_forecasts` aborts
  the run if the shortest-horizon forecast deviates >10% from spot (a 1-day move that
  size never happens here → it signals a bug, not a prediction).
- **Pickle-compat guard.** An unpinned sklearn upgrade once made cached pickles
  unloadable (5-day outage). `load_models()` now validates a `model_version`/`target`
  marker and treats *any* unpickling failure as "no models" → automatic retrain instead
  of a crash.

---

## 7. Validated Track Record

### 7.1 Walk-forward CV benchmark evolution

| Horizon | v1.0 (level target) | v2.0 (log-return) | v2.1 (selection + causal cycle) | v2.2 (+microstructure) |
|---|---|---|---|---|
| 1-day | 5.7% | 2.8% | **2.6%** | 2.6% |
| 7-day | 10.8% | 6.8% | **6.2%** | 6.3% |
| 14-day | 14.1% | 9.9% | 9.2% | **8.8%** |
| 28-day | 16.8% | 9.8% | 9.8% | 9.9% |
| 90-day | 36.6% | 33.5% | **17.6%** | 17.6% |
| Regime (AUC) | 0.575 | 0.660 | **0.778** | 0.778 |

The big jumps: **log-returns** (v2.0, the extrapolation-ceiling fix) and **causal cobweb
features + selection** (v2.1, which halved 90-day MAPE by killing a lookahead leak).
Microstructure (v2.2) helped 14-day specifically.

### 7.2 Live track record (rolling-30, from `track_record.json`, Jun 2026)

| Horizon | MAPE | Hit-rate ≤5% |
|---|---|---|
| 1-day | 2.8% | 90% |
| 3-day | 2.9% | 90% |
| 7-day | 5.4% | 65% |
| 14-day | 5.7% | 55% |
| 28-day | 6.7% | 50% |
| 90-day | 12.9% | 0% |

Short horizons are strong; the 90-day is structurally limited by the ~130-row monthly
sample, not the target definition. **This live-vs-CV gap is exactly why you publish a
track record:** it keeps the model honest and surfaces regime-dependent degradation.

---

## 8. What Works / What Doesn't (Distilled Learnings)

### ✅ What works
- **GBDT + log-returns + walk-forward CV** on a single covariate-rich series. Boring,
  robust, near-SOTA for this data size.
- **Causal feature construction** — the discipline that prevents silent lookahead.
- **Out-of-sample permutation selection** to fight overfitting when features ≫ rows.
- **Matching model capacity to sample size** across horizons.
- **An immutable forecast ledger** — the product moat and the model's conscience.
- **Microstructure from sub-daily detail** — cross-venue dispersion as a leading signal.
- **CSV-seed + API-topup collectors** that degrade gracefully.

### ❌ What doesn't / what we deliberately avoid
- **Deep learning (LSTM, TFT, N-BEATS, PatchTST).** At ~3,000 points on one series they
  overfit and lose to a tuned GBDT; they need cross-series pooling we don't have. A 2026
  literature survey (M4/M5/M6, "Are Transformers Effective?", Monash) confirmed this for
  this exact data regime.
- **Zero-shot foundation models (Chronos/TimesFM/Moirai) as the primary forecaster.**
  Great for cold-start univariate; they cannot synthesise our heterogeneous lagged
  covariates, and fine-tuning on 3,000 points overfits.
- **RL for price prediction.** RL is an MDP/policy solver, not a regressor. (It *is* the
  right tool for a downstream "sell/store" decision — but that is a separate layer, not
  the forecaster.)
- **Price levels as the target.** Caps forecasts at the historical range. See §6.1.
- **Centered-window / two-sided smoothing in features.** Leaks the future.
- **Throwing all 69 features at a small weekly model.** Negative R².

### 🔭 Known open follow-ups
- House-mix-adjusted price index (remove auction-rotation noise from the target itself).
- Collector guard for the rare "sold > arrived" data-entry error.
- **Calibrated conformal prediction intervals** (EnbPI/ACI) across *all* horizons —
  currently only the 90-day has a (Gaussian) interval; the 1–28d emit point forecasts
  only. This is the highest-value, lowest-risk modelling upgrade on the backlog.

---

## 9. Replicating This for Coffee (or any commodity)

The architecture is commodity-agnostic. What changes is **the data sources and the
domain features**, not the pipeline skeleton. Below, "transfers as-is" means copy the
code and just repoint it.

### 9.1 What transfers with zero or trivial change
- The whole **orchestration** (`run_daily.py`), **DB layer** (`db.py`), **assembly**
  merge strategy, **walk-forward CV**, **feature selection**, **log-return target**,
  **model family pattern**, **validation**, **export**, and **static-webapp** approach.
- **T1 Technical/Momentum** and **T2 Calendar/Seasonal** tiers — purely price/date
  driven, identical for coffee.
- The **guardrails** (stale-anchor sanity check, pickle-version marker).

### 9.2 The core question: where does coffee's price come from?

Cardamom has a clean, public, single-market **daily auction** (Spices Board) with
per-auctioneer detail — that is unusually good. Coffee price discovery is different and
you must pick your target series deliberately:

| Option | Source | Pros | Cons |
|---|---|---|---|
| **ICE futures** — Arabica (`KC`), Robusta (`RM`/London) | Yahoo Finance / exchange | Liquid, daily, the global benchmark | Futures ≠ farmgate; roll/contract handling needed |
| **ICO composite & group indicator prices** | International Coffee Organization | Daily, origin-group granularity, free | Indicator not transactable; some lag |
| **Local auction/market** — e.g. India Coffee Board e-auctions, Ethiopia ECX, Kenya NCE | national boards/exchanges | Closest analogue to the cardamom auction; may carry microstructure | Fragmented, less complete history, harder to scrape |

**Recommendation:** pick **one primary target** (most likely ICE Arabica `KC=F` for a
global product, or India Coffee Board e-auction prices if you want the farmgate +
microstructure story that mirrors cardamom). Everything downstream keys off that one
series, exactly as `avg_price` does here.

### 9.3 Feature-tier porting map

| Tier | Cardamom | Coffee equivalent | Notes |
|---|---|---|---|
| **T1 Technical** | auction avg price | chosen coffee price series | as-is |
| **T1b Microstructure** | cross-auctioneer dispersion | only if you use an **auction** target (ECX/NCE/India Coffee Board lots). For ICE futures, replace with **futures microstructure**: term-structure (front–back spread, contango/backwardation), open interest, volume, COT positioning | the *concept* (dispersion/imbalance leads price) transfers; the *fields* differ |
| **T2 Calendar** | Indian festivals, lean/strong season | coffee harvest/flowering calendars **per origin** (Brazil Apr–Sep harvest, Colombia mitaca, Vietnam Robusta Oct–Jan); demand seasonality is weaker | re-author the calendar CSV |
| **T3 Weather/ENSO** | Idukki + Guatemala rainfall, ENSO | **Brazil (Minas Gerais) frost & drought** is the dominant coffee weather driver, plus Vietnam (Central Highlands), Colombia. Keep ENSO — it strongly drives Brazil/Vietnam rainfall and frost risk | **most important tier for coffee.** Add a **frost-degree-day / minimum-temperature** feature for Brazil — coffee's biggest shocks are frosts |
| **T4 Macro** | USD/INR | **USD/BRL** (Brazil real is the key coffee FX), plus crude (freight), and USD/VND for Robusta | repoint FX |
| **T5 Trade/Supply** | Guatemala exports, Saudi imports | **Brazil + Vietnam + Colombia exports**, ICO production/inventory estimates, **certified stocks** (ICE warehouse stocks are a high-frequency supply gauge) | certified stocks are a great, frequently-updated supply signal coffee has that cardamom lacks |
| **T6 Structural/Cycle** | cardamom cobweb (3-yr), MGNREGA cost floor | coffee has a **biennial bearing cycle** (on-year/off-year yield alternation) — encode it; cost floor via origin production cost surveys | the cobweb→biennial mapping is the key structural-feature change |

### 9.4 Things coffee gives you that cardamom doesn't (exploit them)
- **Liquid futures + options** → real implied volatility, term structure, and COT
  positioning as features. This is richer microstructure than a physical auction.
- **ICE certified stocks** → a clean, high-frequency global inventory signal.
- **Two correlated products** (Arabica vs Robusta) → the **arb spread** is itself a
  feature and a potential second series for light cross-series pooling (the one place a
  global/DL model could start to earn its keep — but only with more than one series).

### 9.5 Things to watch / likely failure modes when porting
- **Frost is a jump process.** Brazilian frost events cause discontinuous gaps that no
  smooth feature predicts well. Lean on the conformal intervals (backlog item) and
  treat the regime classifier ("severe drawup/drawdown within N months") as the primary
  risk product, not the point forecast.
- **Futures roll.** If you target futures, build a continuous back-adjusted series
  carefully — a naive splice injects fake returns. Prefer a roll-adjusted continuous
  contract or use the front-month with an explicit roll feature.
- **Sample size still governs capacity.** Coffee has decades of daily futures (~10k+
  rows) — *more* than cardamom, which actually widens the door slightly for richer
  models, but the §8 verdict (GBDT first) still holds until you pool multiple series.
- **Re-tune the feature-selection k and CV windows** in `config.py` to the new sample
  sizes; don't copy cardamom's numbers blindly.

### 9.6 Concrete replication checklist
1. Fork the repo; rename package/paths; keep `pipeline/` structure intact.
2. Decide the **primary price target** (§9.2) and write its collector (scrape/API +
   CSV seed) following `collectors/auction.py`'s pattern.
3. Repoint weather to **Brazil/Vietnam/Colombia**, FX to **USD/BRL**, trade to
   **Brazil/Vietnam exports + ICE certified stocks**; keep ENSO.
4. Re-author the **calendar/harvest CSV** and the **structural cycle** (biennial) and
   **cost-floor** features in T6.
5. Replace auction microstructure with **futures microstructure** (or keep auction
   microstructure if you chose an auction target).
6. Keep **log-return targets, walk-forward CV, permutation selection** unchanged.
7. Re-fit, read the walk-forward MAPE per horizon, and **only then** decide whether any
   horizon justifies more model complexity.
8. Stand up the **forecast ledger + validation loop from day one** — the track record
   is the product.

---

## 10. Running the Pipeline

```bash
pip install -r requirements.txt

# First-time: seed the DB from the bundled CSV/XLS history and train all models
python -m pipeline.run_daily --seed

# Normal daily run (collect → validate → predict → export)
python -m pipeline.run_daily

# Force a full retrain (weekly cadence)
python -m pipeline.run_daily --retrain

# Regenerate JSON exports only
python -m pipeline.run_daily --export-only
```

**Schedule (`crontab.txt`):** daily run at 6 PM IST (after auctions close), full retrain
Sunday, monthly data refresh on the 1st. Output JSON lands in `cardamom_webapp/data/`
and is served statically.

**Stack:** Python 3.9+, scikit-learn, pandas, numpy, joblib, requests. SQLite for
storage, joblib for model serialisation. No GPU, no server, no paid infra beyond the
Comtrade key.

---

## 11. Design Principles to Carry Forward

1. **The pipeline is the product, not the model.** Reliability, causal correctness, and
   the immutable track record matter more than squeezing out the last MAPE point.
2. **Causality is non-negotiable.** Every feature is `.shift(1)`-clean; every monthly
   feed is `merge_asof(backward)`; every CV split is walk-forward with a purge gap.
3. **Match capacity to data.** More rows → more model; never the reverse.
4. **Forecast returns, display prices.** Stationarity beats sophistication.
5. **Validate in public, every day.** A model you can't audit is a model you can't trust.
6. **Degrade, don't crash.** Every external dependency has a fallback and a guardrail.
```
