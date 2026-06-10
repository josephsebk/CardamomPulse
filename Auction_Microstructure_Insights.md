# Auction Microstructure Insights

**Source:** per-auctioneer rows from the Spices Board auction record
(`Small Cardamom Auction Prices.xls`, 5,440 rows, Nov 2014 – Feb 2026,
plus daily scrapes going forward).
**Analysis date:** June 2026 · **Related model release:** v2.2

The forecasting pipeline historically consumed only the daily
volume-weighted average of all auctions. This analysis mined the
per-auctioneer detail underneath — typically two auctions per day, run by
different houses — and found both new model features and market-structure
facts that the daily averages destroy.

---

## 1. A stable ~4% price hierarchy between auction houses

**Method.** Raw per-house average prices are confounded by *when* each house
was active (a house active only in 2015–18 looks cheap because the whole
market was cheap then). The honest comparison is paired: on every day with
two or more auctions, measure each house's average price against that same
day's volume-weighted market average. Houses with ≥100 paired observations:

| House | vs same-day market | Days above market | t-stat |
|---|---|---|---|
| South Indian Green Cardamom Co, Kochi | **+0.87%** | 83% | +14.6 |
| Mas Enterprises, Vandanmettu | +0.73% | 75% | +11.7 |
| Cardamom Planters Mktg. Co-op Society | +0.73% | 71% | +5.0 |
| Idukki Dist. Traditional Producer Co | +0.72% | 69% | +9.5 |
| Vandanmedu Green Gold Producer Co | +0.58% | 77% | +8.9 |
| Kerala Cardamom Processing & Mktg, Thekkady | +0.30% | 58% | +4.2 |
| Sugandhagiri Spices Promoters & Traders | −0.31% | 42% | −5.0 |
| Green House Cardamom Mktg. India | −0.43% | 42% | −3.6 |
| Header Systems (India), Nedumkandam | −0.74% | 28% | −11.4 |
| Green Cardamom Trading Company | −1.33% | 28% | −7.3 |
| Spice More Trading Company, Kumily | −1.35% | 28% | −9.3 |
| Cardamom Growersforever | −1.96% | 27% | −12.5 |
| Cardamom Planters' Assoc., Santhanpara | **−3.05%** | 8% | −19.8 |

The top-to-bottom spread is **~4% — roughly half the 14-day forecast
model's entire error**. Santhanpara prices below the same-day market on 92%
of days (t = −19.8); this is structure, not noise. Likely drivers are lot
quality/grading composition and differing buyer pools, but the premium is
systematic whatever its cause.

**Implication (product):** "where you sell is worth several percent,
consistently" is actionable, durable advice for farmers — arguably worth
more to the audience than an extra point of forecast accuracy.

## 2. The market is a weekly rotation, not a continuous market

Each major house owns a primary weekday (counts of auction rows by day):

| House | Primary day |
|---|---|
| Idukki Dist. Traditional Producer Co | Monday |
| South Indian Green Cardamom Co + Header Systems | Tuesday |
| Sugandhagiri Spices | Wednesday |
| Kerala Cardamom Processing & Mktg | Thursday |
| Cardamom Growersforever | Friday |
| Mas Enterprises + Green House | Saturday |

Consequences:

- The "daily price" is a **rotating composite whose house-mix changes by
  weekday**. Combined with finding 1, part of any day-over-day price change
  is just *which houses auctioned*, not the market moving. The models'
  calendar features (`week_sin`/`week_cos`) have likely been absorbing some
  of this all along.
- **Future improvement:** a house-mix-adjusted price index — correct each
  day's average for the known premia of that day's auctioneers — would
  remove a structural noise source from the forecast target itself.

## 3. Cross-auction price dispersion leads the market — and leans bullish

When the same-day auctions disagree on price, the next two weeks move more,
and skew upward. Quartiles of `disp_cv_ma14` (14-day mean of the
cross-auction price coefficient of variation) vs the subsequent 14-day move:

| Dispersion quartile | Mean abs. 14d move | Mean 14d return |
|---|---|---|
| Q1 (low) | 5.9% | −0.3% |
| Q2 | 7.6% | −0.9% |
| Q3 | 7.8% | +1.2% |
| Q4 (high) | **8.6%** | **+2.8%** |

Interpretation: disagreement between auction houses tends to mark
accumulating demand pressure rather than distribution. This asymmetry is
presumably why `disp_cv_ma14` earned a **top-3 walk-forward permutation
rank in both the 7-day and 14-day models** (it ranked #2 at 14d, where it
cut MAPE from 9.2% to 8.8%).

## 4. Structural churn and data quality

- **Roster turnover:** State Trading Corporation exited in 2018, Vandanmedu
  Green Gold in 2020. A 2024–25 wave of entrants includes
  **South Indian Cardamom Online Auction Pvt Ltd** (e-auctions), RNS Spices,
  Climate Natural Spices, and **Idukki Mahila** (a women's producer
  company). The shift toward online auctions may change microstructure
  dynamics; the `n_auctions` feature tracks participation going forward.
- **"Sold > arrived" anomaly:** only 3 of 5,440 rows. The worst —
  2021-01-02, Green Cardamom Trading, 511,177 kg sold vs 52,095 kg arrived —
  is almost certainly a data-entry error (extra digit). The other two are
  ~10% overruns, plausibly carryover stock. A guard in the collector
  (e.g. cap sold at 1.5× arrived) would stop the bad day from distorting
  volume features.

---

## How the pipeline uses this (model v2.2)

Computed per day in the auction collector before averaging, stored in
`auction_daily`, and exposed as rolling features (Tier 1b, `add_micro`):

| Feature | Definition |
|---|---|
| `price_disp_cv` / `price_range_pct` | CV / max–min range of per-auction avg prices, ÷ day's weighted avg |
| `n_auctions` | auctions held that day |
| `unsold_pct` | 1 − sold/arrived (daily totals) |
| `avg_lot_kg` | arrived kg per lot |

Walk-forward CV impact (v2.1 → v2.2): **14d 9.2% → 8.8% MAPE**; 7d/28d
neutral; 90d unchanged (its selection still prefers cumulative rainfall);
regime unchanged — microstructure is **excluded from the monthly candidate
pool** because monthly averaging destroys these short-lived signals and
including them degraded regime AUC from 0.778 to 0.712.

## Open follow-ups

1. House-mix-adjusted daily price index (removes rotation noise from the
   target).
2. Collector guard for the sold > arrived data error.
3. Surface the house premium table in farmer-facing reporting.
4. Watch whether the 2024+ online-auction entrants compress the house
   premium spread over time.
