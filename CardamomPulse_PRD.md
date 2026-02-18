# CardamomPulse — Product Requirements Document (PRD)

**Version:** 1.0
**Date:** Feb 2026
**Author:** [Founding Team]
**Scope:** PRD + Interactive HTML Mockup (No production data pipeline yet)

---

## 1. Executive Summary

CardamomPulse is a forward-looking price intelligence platform for the global small cardamom market.

Today, farmers, traders, exporters, and procurement managers make irreversible planting, inventory, and contracting decisions using backward-looking charts and intuition.

CardamomPulse provides:

- 7–56 day probabilistic price forecasts
- Market regime detection (supply tight / neutral / oversupply)
- Transparent model track record
- Structural analysis integrating weather, ENSO, and global supply
- Subscriber intelligence reports

The goal is to become:

> The trusted quantitative intelligence layer for the cardamom market.

Phase 1 focuses on transparency and credibility.
Phase 2 evolves toward decision support.
Phase 3 expands into multi-origin spice intelligence.

---

## 2. Problem Statement

The cardamom market suffers from:

### 2.1 Backward-Looking Information
- Historical prices are available (Spices Board daily auction data, Agmarknet)
- Forward expectations are not — decisions are made on intuition and WhatsApp speculation

### 2.2 High Volatility
- Weather-driven supply shocks (ENSO cycles cause ±40% annual swings)
- Export-driven price ceilings (Guatemala supplies ~60% of global cardamom)
- Regime shifts every 2–4 years driven by cobweb planting cycles (3-year gestation lag)

### 2.3 Irreversible Decisions
- Planting decisions lock 7–10 years of capital and land
- Inventory holding creates capital risk in a perishable commodity
- Export contracts expose directional risk on 30–90 day commitments

There is no structured, probabilistic, transparent forward price forecasting platform for cardamom.

---

## 3. Product Vision

**3-Year Vision:** Become the Bloomberg Terminal for the cardamom market.

| Phase | Timeline | Focus |
|-------|----------|-------|
| Phase 1 – Forecast Transparency | 0–12 months | Publish daily probabilistic forecasts, build public track record, establish credibility |
| Phase 2 – Decision Intelligence | 12–24 months | Planting cycle analytics, regime-based strategy guidance, alerts and scenario modeling |
| Phase 3 – Market Infrastructure | 24+ months | API access, multi-spice expansion, institutional subscription base |

---

## 4. Target Users

Users are segmented by **decision type** rather than profession.

### Segment A — Planters (Farmers / Estate Managers)

| | |
|-|-|
| **Decisions** | Plant / replant, expand acreage, hold or sell inventory |
| **Pain Points** | High price volatility, weather uncertainty, no forward guidance |
| **Price Sensitivity** | Medium to high |
| **Primary Tier** | Free / Standard |

### Segment B — Traders & Auction Participants

| | |
|-|-|
| **Decisions** | Buy inventory, hold or liquidate, speculative positioning |
| **Pain Points** | Directional risk, short-term volatility |
| **Price Sensitivity** | Low (value-driven) |
| **Primary Tier** | Professional |

### Segment C — Exporters

| | |
|-|-|
| **Decisions** | Forward contracts, inventory hedging |
| **Pain Points** | Margin compression, supply timing risk |
| **Primary Tier** | Professional |

### Segment D — FMCG Procurement

| | |
|-|-|
| **Decisions** | Raw material locking, budget forecasting |
| **Primary Tier** | Professional (later phase) |

---

## 5. Value Proposition

CardamomPulse is the only platform combining:

- Probabilistic forward forecasts
- Regime detection
- Public track record verification
- Weather + ENSO integration
- Global supply indicators

**Core positioning:**

> Quantitative intelligence for the cardamom market.

Not: "AI predicting prices."

---

## 6. Product Scope (Phase 1 MVP)

### Included
- 8-week forecast (weekly granularity)
- 90-day structural outlook with prediction intervals
- Regime indicator (probabilistic)
- Track record transparency with immutable forecast ledger
- Price archive (2014–present)
- Monthly subscriber report
- Subscription tiers
- Static website + daily JSON export

### Explicitly Not Included (Phase 1)
- Intraday trading signals
- Mobile app (responsive web only)
- Real-time streaming
- API access
- Multi-spice expansion
- Automated alerts

---

## 7. Subscription Model

### Free Tier

**Purpose:** Distribution + trust building

| Feature | Access |
|---------|--------|
| Current price | Full |
| 1–2 week forecast | Directional only (up/down/flat) |
| 3–8 week forecast | Blurred / locked |
| Regime indicator | Traffic light label only |
| Price archive | Last 30 days |
| Track record | Summary metrics |
| Subscriber report | Preview (blurred) |

### Standard — ₹499/month

**Target:** Advanced farmers, small traders

| Feature | Access |
|---------|--------|
| Full 8-week forecast | Price + directional change |
| Regime indicator | Probability + brief explanation |
| Price archive | 2-year rolling window |
| Track record | Detailed by horizon |
| Monthly report | PDF via email |

### Professional — ₹1,999/month

**Target:** Traders, exporters, procurement

| Feature | Access |
|---------|--------|
| Full 8-week + 90-day forecast | With confidence intervals |
| Historical forecast ledger | Timestamped, downloadable |
| CSV download | Archive + forecast history |
| Per-horizon accuracy breakdown | Full detail |
| Regime history analytics | Probability timeline + drivers |
| Extended track record | All horizons, all time |

**Future:** Annual subscription option

---

## 8. Page-by-Page Specification

### Page 1 — Dashboard (Public)

**KPI Strip**
- Spot Price (today's weighted-average auction price, daily change)
- 14-Day Expected Range
- Regime Status (Tight / Neutral / Oversupply)
- Model Confidence Level

**8-Week Forecast Fan Chart**
- Center line: point forecast
- Inner band: ±1 MAE (from walk-forward CV history)
- Outer band: ±1 MAPE × price
- Weeks 3–8 blurred for free tier
- 90-day point shown separately with model-native Bayesian prediction interval

**90-Day Price History**
- Area chart with volume overlay

**Regime Timeline**
- Probability strip (color-coded green/amber/red)
- Hover explanation of key drivers

---

### Page 2 — Price Archive

- Full history chart (Nov 2014–present), zoomable
- Searchable table with pagination
- Download CSV (Professional only)
- Granularity toggle: Daily / Weekly / Monthly

---

### Page 3 — Track Record

**Summary Metrics**
- MAPE by horizon
- Directional accuracy
- Total predictions made
- Rolling 30-day accuracy

**Charts**
- Predicted vs actual (toggleable by horizon)
- Accuracy by regime phase
- Accuracy by ENSO phase

**Forecast Ledger**
- Timestamped, immutable prediction log
- Every forecast ever made, with outcome once validated
- Publicly archived (detailed access for paid tiers)

---

### Page 4 — Subscriber Report (Preview)

Sections (blurred for free tier):
- Executive summary
- Forecast table (all horizons)
- Key price drivers this week
- Weather outlook (Idukki rainfall + Guatemala)
- ENSO status and 12-month lag implications
- Regime analysis with probability update

---

## 9. Data Pipeline Architecture (Future State)

### 9.1 Data Sources

The forecasting models are built on 7 external data feeds spanning 5 source types:

| Source | Feed | Frequency | API / Method | Lag |
|--------|------|-----------|-------------|-----|
| **Spices Board of India** | Daily auction prices (avg price, max price, volume, lots by auctioneer) | Daily | HTML scrape from indianspices.com | Same day |
| **Open-Meteo** | Idukki, Kerala weather (rain, temp, humidity) | Daily | Free REST API, no key | Same day |
| **Open-Meteo** | Cobán, Guatemala weather (rain, temp) | Daily | Free REST API, no key | Same day |
| **Yahoo Finance** | USD/INR, Brent Crude, Gold, Nifty 50 | Daily | yfinance Python library | Same day |
| **NOAA CPC** | ENSO Oceanic Niño Index (ONI anomaly) | Monthly | ASCII text file download | ~2 weeks |
| **UN Comtrade** | Guatemala cardamom exports (HS 0908: qty, value, unit price) | Monthly | REST API (subscription key) | ~30 days |
| **UN Comtrade** | Saudi Arabia cardamom imports (HS 0908: qty, value) | Monthly | REST API (subscription key) | ~30 days |
| **Google Trends** | Search interest: "cardamom price", "cardamom plantation", etc. | Monthly | pytrends (unofficial) | ~7 days |
| **Static** | Indian festival/season calendar (wedding, harvest, Eid, Diwali, Onam, Christmas) | Yearly | Pre-generated CSV | None |

**Total: ~15 CSV/data files currently, 9 automated feeds in production.**

### 9.2 Feature Engineering (6 Tiers)

Features are organized in tiers, with different horizons consuming different tier subsets:

| Tier | Features | Used By | Examples |
|------|----------|---------|----------|
| **T1: Technical/Momentum** | 33 per granularity | All horizons | Price lags (1-30d), MAs (7-90d), RSI(14), Bollinger position, volume ratios |
| **T2: Calendar/Seasonal** | 13 daily, 6 monthly | All horizons | Month/week cyclical encoding (sin/cos), festival flags, lean/strong season |
| **T3: Weather/ENSO** | 14 | Weeks 3-8, 90-day, Regime | Idukki cumulative rainfall (28-182d), Guatemala rainfall, ENSO current + lag 6m/12m |
| **T4: Macro** | 6 | Weeks 3-8 | USD/INR level + % change, crude/gold/nifty 28-day % change (not levels) |
| **T5: Trade/Supply** | 5 | Weeks 3-8, 90-day, Regime | Guatemala 3-month rolling export qty + YoY, Saudi import qty + YoY, Guatemala unit price |
| **T6: Structural/Cycle** | 6 | 90-day, Regime | Cost floor (MGNREGA wage × 1.7 × 200 / 0.62 / 350), cobweb cycle position, Google Trends 12m MA |

**Key lesson from prior modeling work:** Using 69 features on 534 weekly rows caused overfitting (negative R²). Production models cap at 15–20 features per horizon via walk-forward feature selection.

### 9.3 Model Architecture

| Horizon | Granularity | Input Rows | Model | Features | Uncertainty |
|---------|-------------|-----------|-------|----------|-------------|
| Weeks 1–2 | Daily (~3,000 rows) | T1 + T2 (38–46 features) | Gradient Boosting Regressor | Technical + calendar | MAPE-derived error bands |
| Weeks 3–8 | Weekly (~550 rows) | T1–T5 (up to 71 features, selected down to ~40) | Stacked GBM + Ridge Regression | All except structural | MAPE-derived error bands |
| 90-day | Monthly (~136 rows) | T1 (subset) + T3 + T5 + T6 (28 features) | Bayesian Ridge Regression | Structural + ENSO + trade | Model-native 80% prediction intervals |
| Regime | Monthly (~130 rows) | T1 (subset) + T3 + T5 + T6 (35 features) | Gradient Boosting Classifier | Structural + ENSO + trade | Calibrated probability (0–100%) |

**Walk-forward cross-validation** throughout — no random splits. Expanding window with purge gaps to prevent lookahead bias.

### 9.4 Daily Pipeline Sequence

```
6:00 PM IST (after auctions close ~5 PM):

1. COLLECT — Scrape today's auction results + fetch weather/financial data
2. STORE — Upsert into SQLite database
3. VALIDATE — Compare yesterday's predictions against today's actual price
4. PREDICT — Run all models on latest features → 8 weekly + 90-day + regime
5. ARCHIVE — Store predictions with timestamp in forecast ledger (immutable)
6. EXPORT — Generate JSON files for website consumption
7. DEPLOY — Push JSON to static hosting (Vercel/Cloudflare)
```

**Weekly (Sunday 2 AM IST):** Full model retrain on latest data. Walk-forward CV metrics logged.

**Monthly (1st of month):** Fetch ENSO, trade data, Google Trends. Regenerate festival calendar if needed.

### 9.5 Merge Strategy

Trade data (monthly, ~30-day reporting lag) is merged using `merge_asof(direction='backward')` to prevent lookahead — each row only sees the most recently available trade data at that point in time, not future reports.

Financial data gaps (weekends, holidays) are forward-filled. ENSO seasonal data is mid-month anchored and forward-filled to daily.

---

## 10. Technical Architecture (MVP)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Frontend | Static HTML + Tailwind CSS CDN + Chart.js | Simple, fast, no build step |
| Hosting | Vercel / Cloudflare Pages | Free tier, global CDN |
| Data delivery | Static JSON files updated daily | No backend server needed |
| Forecast engine | Python 3.9+ with scikit-learn, pandas, numpy | Existing proven models |
| Data store | SQLite (single file) | No server, easy backup, sufficient for batch workload |
| Model serialization | joblib | Standard for scikit-learn models |
| Auth/payments | Stripe Checkout (Phase 2) | Handles Indian cards, UPI support |
| Scheduling | cron on VPS | Simple, reliable for daily batch |

**No backend server for MVP.** Pipeline runs as cron job, exports JSON, served statically.

---

## 11. Success Metrics

### 6-Month Targets

| Metric | Target |
|--------|--------|
| Free users | 500 |
| Standard subscribers | 50 |
| Professional subscribers | 15 |

### Product Quality Metrics

| Metric | Target |
|--------|--------|
| Directional accuracy (all horizons) | >55% |
| 2-week MAPE | <8% |
| 4-week MAPE | <12% |
| Model update latency | <24 hours |

### Business Metrics

| Metric | Target |
|--------|--------|
| Free → paid conversion | 15–20% |
| Monthly churn | <5% |

---

## 12. Risks

### Technical
| Risk | Mitigation |
|------|------------|
| Model overfitting | Walk-forward CV; public track record forces honesty |
| Regime misclassification | Publish as probability, never binary claim |
| Auction scraper breaks (HTML structure change) | Monitoring + manual XLS upload fallback |

### Market
| Risk | Mitigation |
|------|------------|
| Skepticism toward forecasts | Free tier + transparent track record builds trust |
| Farmer unwillingness to pay | Start with traders/exporters (lower price sensitivity) |

### Strategic
| Risk | Mitigation |
|------|------------|
| Large trader builds internal model | Public track record is a moat they can't replicate without publishing |

---

## 13. Competitive Landscape

| Alternative | What It Offers | What It Lacks |
|-------------|---------------|---------------|
| indianspices.com | Official daily auction data | No analytics, no forecasts |
| Agmarknet | Government price portal | Delayed, no forward view |
| Bloomberg/Reuters | Global commodities | No Indian cardamom coverage |
| WhatsApp groups | Trader sentiment | Anecdotal, unstructured |
| Internal trader models | Custom analytics | Not transparent, not verifiable |

**No public forward probabilistic forecast platform exists for cardamom.**

---

## 14. Strategic Moat

Not just forecast accuracy. Accuracy can be replicated.

The moat is the **compounding trust system:**

- **Public forecast ledger** — every prediction timestamped, immutable, verifiable
- **Regime framework** — structural understanding (ENSO, cobweb cycles, cost floor) that survives model changes
- **Multi-source data integration** — 7+ automated feeds from weather, trade, macro, search data
- **Historical archive** — 11+ years of curated daily auction data
- **Track record depth** — after 12 months, thousands of validated predictions

**Trust compounds over time.**

---

## 15. Roadmap (12 Months)

| Quarter | Milestones |
|---------|------------|
| **Q1** | PRD + mockup, data pipeline prototype, public beta with forecast ledger |
| **Q2** | Live daily forecasts, free tier launch, track record accumulation begins |
| **Q3** | Paid subscriptions (Stripe), monthly PDF reports, regime analytics expansion |
| **Q4** | 6-month track record published, institutional outreach, accuracy reporting upgrades |

---

## 16. Long-Term Expansion

- Multi-spice coverage (pepper, turmeric, cloves)
- API access for institutional clients
- Risk dashboards and scenario modeling
- Forward contract advisory tools
- Malayalam language support

---

## Conclusion

CardamomPulse transforms cardamom price discovery from reactive to probabilistic.

It creates a new category: **forward agricultural intelligence for niche commodities.**

The platform starts simple — transparent forecasts + regimes.

It compounds into — market infrastructure.

---

## Appendix: Validated Model Benchmarks

From the existing forecasting framework (walk-forward CV, Feb 2026):

| Horizon | Walk-Forward MAPE | First Live Prediction | Actual | Error |
|---------|-------------------|----------------------|--------|-------|
| 7-day | 11.4% | ₹2,455 (Feb 13) | ₹2,476 | **0.8%** |
| 14-day | 15.0% | ₹2,420 (Feb 20) | ~₹2,440 | **~0.8%** |
| 28-day | 12.1% | ₹2,427 (Mar 6) | Pending | — |
| 90-day | 33.9% | ₹2,096 (May 7) | Pending | — |
| Regime (AUC) | 0.575 | 3% bear probability | — | — |

The live validation (0.8% error on 7-day) significantly outperformed the walk-forward CV average (11.4%), suggesting the model performs well in low-volatility regimes.

---

*End of PRD*
