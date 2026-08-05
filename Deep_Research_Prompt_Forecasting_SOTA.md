# Deep Research Prompt — State of the Art in Agricultural Auction Price Forecasting

Use the prompt below with a deep-research tool. It is written to be self-contained:
it carries our measured diagnostics so the research is targeted at our actual failure
modes rather than returning generic "how to forecast prices" material.

---

## PROMPT

You are researching the state of the art in **short-to-medium horizon price forecasting
for thinly-traded physical agricultural commodities sold at daily auction**, in order to
advise a working forecasting system that has measurably failed to beat a naive baseline.

### System under review

- **Target:** daily volume-weighted average auction price of Indian small cardamom
  (Alleppey Green), from Spices Board e-auctions. ~5.14 auction sessions per week,
  ~3,150 observations since Nov 2014. Prices are the mean of whatever lots actually
  cleared that session, so day-to-day lot-quality composition shifts are embedded in
  the series.
- **Horizons:** 1, 2, 3, 4, 5, 6, 7, 14, 28, 90 days.
- **Current models:** GradientBoostingRegressor for 1–14d; a GBM+Ridge stack on weekly
  bars for 28d; BayesianRidge on monthly bars for 90d; a GradientBoostingClassifier for
  a 6-month "bear regime" flag. Targets are log returns, reconstructed to price levels as
  `anchor x exp(predicted_return)` where the anchor is the last observed auction price.
  Feature selection is walk-forward permutation importance, top-k (k=5 to 20).
- **Features:** own-price lags and moving averages, auction microstructure
  (cross-auctioneer price dispersion, unsold share, auction count, lot size), Idukki and
  Guatemala weather, ENSO/ONI, Guatemala and Saudi/UAE trade flows, USD/INR, crude, gold,
  Nifty, Google Trends, and a festival/harvest calendar.

### Measured diagnostics (378 matured out-of-sample forecasts, Feb–Aug 2026)

1. The models beat a random-walk ("no change") baseline at only 2 of 10 horizons, by
   negligible margins. Diebold-Mariano is insignificant at every horizon except 90 days,
   where the model is **significantly worse** than naive (p < 0.001).
2. Pooled correlation between predicted and realized log return is **-0.064**. Pooled
   directional accuracy is **50.0%**.
3. At 28 and 90 days, **100% of forecasts fell below the actual** during a +24.8% rally
   (mean signed error -₹277 and -₹566/kg). At 90 days the predicted/realized return
   correlation is nevertheless **+0.565**, i.e. relative information exists but a large
   constant negative bias destroys it — a calibration failure, not absent signal.
4. The series is **not** a random walk: lag-1 return autocorrelation is **-0.129**
   (~7 standard errors, n=3150), and variance ratios are **0.69–0.87** across q = 2…60,
   indicating persistent multi-scale mean reversion.
5. A **single-feature linear rule** — regress the h-ahead log return on
   (log price - its 20-session moving average), coefficients fit only on pre-2026 data —
   beats the naive baseline at **all 10 horizons** and the full ML stack at **9 of 10**,
   improving on the ML models by +23% (28d) and +34% (90d) in MAE.
6. Reported price-level MAPE flatters the system: the anchor supplies most of the
   accuracy. At 7 days the mean absolute realized move is 4.65% while published MAPE is
   5.21%.

### What to find

Prioritise peer-reviewed literature, forecasting-competition findings, and central
bank / FAO / USDA / World Bank methodological work. Prefer sources that report results
against naive benchmarks. For each area, report what is established, what is contested,
what the measured effect sizes are, and how well it transfers to a thin daily auction
market with ~3,000 observations.

1. **Benchmarking and evaluation discipline.** Best practice for establishing forecast
   skill in commodity prices: random walk vs drift vs seasonal-naive baselines, skill
   scores, Theil's U, Diebold-Mariano and its small-sample corrections (Harvey-Leybourne-
   Newbold), Clark-West for nested models, forecast encompassing, model confidence sets,
   multiple-testing correction across horizons. What is the standard evidentiary bar for
   claiming skill, and how common is failure to clear it?

2. **Mean reversion in commodity prices.** Theory of storage, convenience yield,
   cost-of-carry; Ornstein-Uhlenbeck and Schwartz one/two/three-factor models;
   Schwartz-Smith short-term/long-term decomposition. How is the speed of mean reversion
   estimated robustly, and how should it inform a term structure of horizons? Which
   specifications forecast best out-of-sample as opposed to fitting well in-sample?

3. **Spurious vs tradeable mean reversion.** How much of measured negative
   autocorrelation in an auction average price is a microstructure artifact — bid-ask
   bounce, non-synchronous trading, and especially **changing lot-quality composition in
   a volume-weighted average of heterogeneous lots**? Methods to separate the artifact
   from genuine economic reversion: quality-adjusted / hedonic price indices, repeat-sales
   indices, matched-lot indices, filtering for composition drift. This is a central
   question for us — how would we test whether our -0.129 lag-1 autocorrelation is real?

4. **Why ML underperforms simple models in low-signal price settings.** Evidence from
   the M4/M5 competitions and the financial ML literature on when gradient boosting and
   deep models beat vs lose to linear and naive methods; sample-size thresholds;
   regularisation and shrinkage; combining/averaging forecasts; the empirical strength of
   simple combinations. What does the evidence say about ~3,000 observations with a low
   signal-to-noise ratio?

5. **Overfitting in model selection itself.** Walk-forward permutation importance with
   top-k selection on small samples; selection bias and how to correct it; purging and
   embargoing (Lopez de Prado); combinatorial purged cross-validation; deflated Sharpe /
   probability of backtest overfitting as analogues for forecast evaluation.

6. **Horizon alignment.** Correct handling of irregular trading calendars when the target
   is defined in calendar days but the data is indexed by trading sessions — a mismatch we
   have confirmed in our own pipeline. Established treatments for irregularly-spaced
   financial time series and missing-session alignment.

7. **Bias correction and recalibration.** Post-hoc recalibration of systematically biased
   forecasts: intercept correction, Theil's correction, rolling bias adjustment, and
   trend-following corrections in a persistent regime. When does bias correction help vs
   overfit? Relevant given our 100% one-sided long-horizon errors.

8. **Probabilistic forecasting.** Moving from point forecasts to calibrated intervals and
   densities: quantile regression, quantile gradient boosting, conformal prediction and
   its time-series variants (EnbPI, adaptive conformal inference), CRPS and pinball loss,
   PIT-based calibration testing, coverage under regime change. For a farmer deciding when
   to sell, what is the evidence that intervals or scenarios beat point forecasts in
   decision value?

9. **Regime switching and structural breaks.** Markov-switching, threshold and smooth-
   transition autoregressions, and break detection for commodity price regimes; forecasting
   performance across breaks. Our market is in a supply-shock regime (Guatemala thrips
   crisis) that dominates recent behaviour.

10. **Fundamentals and exogenous drivers — do they actually add out-of-sample value?**
    Evidence on weather, ENSO/ONI, trade flows, exchange rates and search-trend data
    improving *out-of-sample* agricultural price forecasts, with attention to publication
    bias and to the frequency mismatch between slow-moving monthly/annual predictors and
    daily price targets. At which horizons do fundamentals begin to pay?

11. **Spice and Indian agricultural market specifics.** Any literature on cardamom,
    pepper, or Indian agricultural auction/mandi price forecasting; Spices Board auction
    microstructure; the Guatemala-India global cardamom supply relationship; APMC/mandi
    price modelling and government price-forecasting programmes.

12. **Decision-relevant framing.** Literature on what actually helps smallholder farmers:
    optimal selling/storage timing under uncertainty, price-forecast decision value,
    whether published point forecasts improve or harm farmer outcomes, and evaluation
    frameworks measuring economic value rather than statistical error.

### Deliverable

Produce a structured report covering:

- A ranked synthesis of methods most likely to beat a random walk on **this specific
  problem**, with expected effect sizes and the evidence behind them.
- An explicit verdict on whether short-horizon (1–7 day) forecasting of this series is
  plausibly a near-martingale problem where no method should be expected to add value,
  and at what horizon genuine predictability is documented to begin.
- A recommended evaluation protocol we should adopt as the standard of proof.
- A diagnosis of whether our mean-reversion finding is likely genuine or a
  composition/microstructure artifact, and the specific tests that would settle it.
- Key papers with full citations, and any public code or benchmark datasets.
- Explicit flags where evidence is weak, contested, or likely subject to publication bias.
