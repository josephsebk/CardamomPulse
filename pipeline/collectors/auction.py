"""Collect daily cardamom auction prices from Spices Board / XLS fallback.

v3.0 additions:
- Outlier guard capping sold_kg at AUCTION_SOLD_TO_ARRIVED_MAX_RATIO * arrived_kg
  (protects volume features from data-entry errors like 2021-01-02).
- Strictly causal expanding-window house-mix-adjusted daily price index
  (`adj_price`, `house_effect_log`) that strips out weekday auctioneer rotation
  noise (~4% permanent spread across auction houses) without any lookahead.
- Persists latest causal house premium hierarchy into `auction_house_effects`.
"""

import logging
import math
import re
import numpy as np
import pandas as pd
import requests
from pipeline.config import (
    AUCTION_SOLD_TO_ARRIVED_MAX_RATIO,
    CSV_DIR,
    HOUSE_EFFECT_HALFLIFE_DAYS,
    HOUSE_EFFECT_SHRINKAGE_N,
)
from pipeline.db import get_conn

log = logging.getLogger(__name__)

SPICES_BOARD_URL = "https://www.indianspices.com/marketing/auction/small-cardamom.html"

_TICKER_RE = re.compile(
    r"Spice:\s*Small Cardamom[^,]*,\s*"
    r"Date of Auction:\s*([\w\d -]+?)\s*,\s*"
    r"Auctioneer:\s*(.+?)\s*,\s*"
    r"No\.of lots:\s*([\d.]+)\s*,\s*"
    r"Qty Arrived \(Kgs\):\s*([\d.]+)\s*,\s*"
    r"Qty Sold \(Kgs\):\s*([\d.]+)\s*,\s*"
    r"Max Price \(Rs\./Kg\):\s*([\d.]+)\s*,\s*"
    r"Avg\. Price \(Rs\./Kg\):\s*([\d.]+)",
    re.DOTALL,
)


def _normalize_auctioneer(name: str) -> str:
    """Normalize auctioneer name whitespace/punctuation for stable fixed effects."""
    s = re.sub(r"\s+", " ", str(name or "").strip())
    return s


def _sanitize_raw_auctions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data quality guards before aggregation."""
    clean = df.copy()
    clean = clean.dropna(subset=["Date", "AvgPrice"])
    clean = clean[clean["AvgPrice"] > 0].copy()

    if "Auctioneer" in clean.columns:
        clean["Auctioneer"] = clean["Auctioneer"].map(_normalize_auctioneer)

    # Guard against data-entry typos where sold >> arrived (e.g. 2021-01-02:
    # 511,177 kg sold vs 52,095 kg arrived).
    if "Qty_Arrived_Kg" in clean.columns and "Qty_Sold_Kg" in clean.columns:
        cap = clean["Qty_Arrived_Kg"] * AUCTION_SOLD_TO_ARRIVED_MAX_RATIO
        bad = (clean["Qty_Arrived_Kg"] > 0) & (clean["Qty_Sold_Kg"] > cap)
        n_bad = int(bad.sum())
        if n_bad > 0:
            log.warning(
                f"Capping {n_bad} auction row(s) with sold > "
                f"{AUCTION_SOLD_TO_ARRIVED_MAX_RATIO:.1f}x arrived"
            )
            clean.loc[bad, "Qty_Sold_Kg"] = cap.loc[bad]

    # Guard against negative/zero max price
    if "MaxPrice" in clean.columns:
        bad_max = clean["MaxPrice"].notna() & (clean["MaxPrice"] < clean["AvgPrice"])
        if bad_max.any():
            clean.loc[bad_max, "MaxPrice"] = clean.loc[bad_max, "AvgPrice"]

    return clean


def scrape_auction_page() -> pd.DataFrame | None:
    """Scrape today's auction data from indianspices.com ticker text."""
    try:
        resp = requests.get(SPICES_BOARD_URL, timeout=30, headers={
            "User-Agent": "CardamomPulse/1.0 (price-research)"
        })
        resp.raise_for_status()

        clean = re.sub(r"<[^>]+>", "", resp.text)
        clean = re.sub(r"\s+", " ", clean)

        matches = _TICKER_RE.findall(clean)
        if not matches:
            log.warning("No Small Cardamom entries found in page ticker")
            return None

        rows = []
        for m in matches:
            rows.append({
                "Date": pd.to_datetime(m[0].strip(), format="%d-%b-%Y", dayfirst=True),
                "Auctioneer": m[1].strip(),
                "Lots": float(m[2]),
                "Qty_Arrived_Kg": float(m[3]),
                "Qty_Sold_Kg": float(m[4]),
                "MaxPrice": float(m[5]),
                "AvgPrice": float(m[6]),
            })
        df = pd.DataFrame(rows)
        log.info(f"Scraped {len(df)} Small Cardamom entries from Spices Board ticker")
        return df
    except Exception as e:
        log.warning(f"Spices Board scrape failed: {e}")
        return None


def load_xls_fallback() -> pd.DataFrame:
    """Load the static XLS file with historical auction data."""
    xls_path = CSV_DIR / "Small Cardamom Auction Prices.xls"
    if not xls_path.exists():
        raise FileNotFoundError(f"Auction XLS not found: {xls_path}")

    df = pd.read_html(str(xls_path), header=0)[0]
    df.columns = [
        "Date", "Auctioneer", "Lots", "Qty_Arrived_Kg",
        "Qty_Sold_Kg", "MaxPrice", "AvgPrice",
    ]
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
    for col in ["Lots", "Qty_Arrived_Kg", "Qty_Sold_Kg", "MaxPrice", "AvgPrice"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "AvgPrice"])
    return df


def compute_causal_house_effects(
    df: pd.DataFrame,
    halflife_days: float = HOUSE_EFFECT_HALFLIFE_DAYS,
    shrinkage_n: float = HOUSE_EFFECT_SHRINKAGE_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strictly causal auctioneer fixed effects and house-adjusted prices.

    For each trading date `t` in chronological order:
      1. Look up each active house's *prior* exponentially-weighted paired
         log-deviation from same-day market average (strictly from dates < t),
         shrunk toward 0 via empirical Bayes: `alpha_h = w_sum / (w_cnt + shrinkage_n)`.
      2. Compute date `t`'s volume-weighted house composition effect:
         `E_t = sum(w_{h,t} * alpha_h)` and denoised index:
         `adj_price_t = avg_price_t * exp(-E_t)`.
      3. ONLY AFTER scoring date `t`, if date `t` has >= 2 auctions, update the
         exponentially-weighted paired log-deviation state for each house.

    Returns:
      - daily_effects DataFrame: [Date, house_effect_log, adj_price]
      - house_summary DataFrame: [auctioneer, premium_pct, n_obs]
    """
    if df.empty or "Auctioneer" not in df.columns:
        return pd.DataFrame(columns=["Date", "house_effect_log", "adj_price"]), pd.DataFrame(
            columns=["auctioneer", "premium_pct", "n_obs"]
        )

    decay_rate = math.log(2.0) / max(float(halflife_days), 1.0)
    # Per-house state: [ewm_sum_log_dev, ewm_weight_sum, last_date_ord, raw_count]
    state: dict[str, list] = {}

    records = []
    for dt, grp in df.sort_values("Date").groupby("Date", sort=True):
        ord_day = dt.toordinal()
        sold = grp["Qty_Sold_Kg"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        prices = grp["AvgPrice"].to_numpy(dtype=float)
        houses = grp["Auctioneer"].to_numpy()

        total_sold = float(sold.sum())
        weights = sold / total_sold if total_sold > 0 else np.full(len(grp), 1.0 / len(grp))
        wavg = float(np.sum(prices * weights))

        # 1. Estimate today's house-mix effect strictly from prior days (< t)
        day_effect = 0.0
        for h, w in zip(houses, weights):
            st = state.get(h)
            if st is not None:
                ewm_sum, ewm_w, last_ord, _ = st
                dt_gap = max(0, ord_day - last_ord)
                decay = math.exp(-decay_rate * dt_gap)
                eff_sum = ewm_sum * decay
                eff_w = ewm_w * decay
                alpha_h = eff_sum / (eff_w + shrinkage_n)
                day_effect += float(w) * alpha_h

        # Bound single-day adjustment to a safe ±6% band
        day_effect = float(np.clip(day_effect, -0.06, 0.06))
        adj_price = wavg * math.exp(-day_effect)
        records.append({
            "Date": dt,
            "house_effect_log": day_effect,
            "adj_price": adj_price,
        })

        # 2. Update house fixed-effect state using paired same-day deviations
        if len(grp) >= 2 and wavg > 0:
            for h, p in zip(houses, prices):
                if p <= 0:
                    continue
                log_dev = float(np.clip(math.log(p / wavg), -0.15, 0.15))
                st = state.get(h)
                if st is None:
                    state[h] = [log_dev, 1.0, ord_day, 1]
                else:
                    ewm_sum, ewm_w, last_ord, raw_n = st
                    decay = math.exp(-decay_rate * max(0, ord_day - last_ord))
                    state[h] = [
                        ewm_sum * decay + log_dev,
                        ewm_w * decay + 1.0,
                        ord_day,
                        raw_n + 1,
                    ]

    # Build final house premium summary table (at latest date)
    latest_ord = df["Date"].max().toordinal()
    summary_rows = []
    for h, (ewm_sum, ewm_w, last_ord, raw_n) in state.items():
        if raw_n < 15:
            continue
        decay = math.exp(-decay_rate * max(0, latest_ord - last_ord))
        alpha_h = (ewm_sum * decay) / (ewm_w * decay + shrinkage_n)
        summary_rows.append({
            "auctioneer": h,
            "premium_pct": round((math.exp(alpha_h) - 1.0) * 100.0, 2),
            "n_obs": int(raw_n),
        })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("premium_pct", ascending=False).reset_index(drop=True)

    return pd.DataFrame(records), summary_df


def aggregate_daily(df: pd.DataFrame, return_house_effects: bool = False):
    """Aggregate per-auctioneer rows to daily totals, microstructure signals,
    and the strictly causal house-mix-adjusted daily price index."""
    clean = _sanitize_raw_auctions(df)

    def _wavg(group):
        sold = group["Qty_Sold_Kg"].fillna(0.0).clip(lower=0.0)
        weights = sold / sold.sum() if sold.sum() > 0 else 1 / len(sold)
        wavg = float((group["AvgPrice"] * weights).sum())
        arrived = float(group["Qty_Arrived_Kg"].sum())
        lots = float(group["Lots"].sum())
        max_p = float(group["MaxPrice"].max())
        n = len(group)
        if n > 1 and wavg > 0:
            disp_cv = float(group["AvgPrice"].std(ddof=0) / wavg)
            range_pct = float((group["AvgPrice"].max() - group["AvgPrice"].min()) / wavg)
        else:
            disp_cv, range_pct = 0.0, 0.0
        grade_spread = (max_p / wavg - 1.0) if (wavg > 0 and pd.notna(max_p)) else np.nan
        return pd.Series({
            "Lots": lots,
            "Qty_Arrived_Kg": arrived,
            "Qty_Sold_Kg": float(sold.sum()),
            "AvgPrice": wavg,
            "MaxPrice": max_p,
            "NAuctions": float(n),
            "PriceDispCV": disp_cv,
            "PriceRangePct": range_pct,
            "UnsoldPct": max(0.0, 1.0 - float(sold.sum()) / arrived) if arrived > 0 else np.nan,
            "AvgLotKg": arrived / lots if lots > 0 else np.nan,
            "GradeSpreadPct": grade_spread,
        })

    daily = clean.groupby("Date").apply(_wavg, include_groups=False).reset_index()

    # Attach strictly causal house-mix-adjusted daily index
    effects_df, house_summary = compute_causal_house_effects(clean)
    if not effects_df.empty:
        daily = daily.merge(effects_df, on="Date", how="left")
    else:
        daily["house_effect_log"] = 0.0
        daily["adj_price"] = daily["AvgPrice"]

    daily = daily.rename(columns={
        "Date": "date",
        "Lots": "lots",
        "Qty_Arrived_Kg": "arrived_kg",
        "Qty_Sold_Kg": "sold_kg",
        "AvgPrice": "avg_price",
        "MaxPrice": "max_price",
        "NAuctions": "n_auctions",
        "PriceDispCV": "price_disp_cv",
        "PriceRangePct": "price_range_pct",
        "UnsoldPct": "unsold_pct",
        "AvgLotKg": "avg_lot_kg",
        "GradeSpreadPct": "grade_spread_pct",
    })
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily = daily.sort_values("date").reset_index(drop=True)

    if return_house_effects:
        return daily, house_summary
    return daily


def collect_auction() -> pd.DataFrame:
    """Main entry: load XLS history, merge with fresh scrape, aggregate, upsert."""
    raw = load_xls_fallback()
    log.info(f"Loaded XLS fallback: {len(raw)} rows")

    scraped = scrape_auction_page()
    if scraped is not None and len(scraped) > 0:
        raw = pd.concat([raw, scraped], ignore_index=True)
        raw = raw.drop_duplicates(subset=["Date", "Auctioneer"], keep="last")
        log.info(f"Merged {len(scraped)} scraped rows with XLS history")

    daily, house_summary = aggregate_daily(raw, return_house_effects=True)

    conn = get_conn()
    cols = list(daily.columns)
    sql = (f"INSERT OR REPLACE INTO auction_daily ({', '.join(cols)}) "
           f"VALUES ({', '.join(['?'] * len(cols))})")
    conn.executemany(sql, daily.values.tolist())

    if not house_summary.empty:
        conn.executemany(
            """INSERT OR REPLACE INTO auction_house_effects
               (auctioneer, premium_pct, n_obs, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            house_summary[["auctioneer", "premium_pct", "n_obs"]].values.tolist(),
        )
        log.info(f"Updated causal house premium hierarchy ({len(house_summary)} houses)")

    conn.commit()
    conn.close()

    return daily
