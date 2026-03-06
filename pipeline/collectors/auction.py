"""Collect daily cardamom auction prices from Spices Board / XLS fallback."""

import json
import logging
import re
import pandas as pd
import requests
from pipeline.config import CSV_DIR
from pipeline.db import get_conn, get_latest_date

log = logging.getLogger(__name__)

SPICES_BOARD_URL = "https://www.indianspices.com/marketing/auction/small-cardamom.html"
DAILY_PRICE_URL = "https://www.indianspices.com/marketing/price/domestic/daily-price-small.html"

# Regex to extract the auction_array1 JSON from the daily-price page's
# exportExcel() function.  The array is assigned as a JS literal.
_AUCTION_ARRAY_RE = re.compile(
    r"var\s+auction_array1\s*=\s*(\[.*?\])\s*;",
    re.DOTALL,
)

# Regex to parse the ticker text format on the Spices Board page.
# Each Small Cardamom entry looks like:
#   Spice: Small Cardamom</b>,  Date of Auction: 20-Feb-2026,
#   Auctioneer: ...,  No.of lots: 245,  Qty Arrived (Kgs): 72074.4,
#   Qty Sold (Kgs): 70270.6,  Max Price (Rs./Kg): 2912.00,
#   Avg. Price (Rs./Kg): 2415.68
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


def fetch_daily_price_data() -> pd.DataFrame | None:
    """Fetch auction records from the Spices Board daily-price page.

    The page embeds a full JSON array (auction_array1) inside its
    exportExcel() JS function, containing all historical auction rows.
    This is the same data behind the "Export Excel" button.
    """
    try:
        resp = requests.get(DAILY_PRICE_URL, timeout=60, headers={
            "User-Agent": "CardamomPulse/1.0 (price-research)"
        })
        resp.raise_for_status()

        m = _AUCTION_ARRAY_RE.search(resp.text)
        if not m:
            log.warning("Could not find auction_array1 in daily-price page")
            return None

        records = json.loads(m.group(1))
        if not records:
            log.warning("auction_array1 is empty")
            return None

        df = pd.DataFrame(records)
        df = df.rename(columns={
            "auction_date": "Date",
            "auctioneer": "Auctioneer",
            "no_of_lots": "Lots",
            "total_qty_arrived": "Qty_Arrived_Kg",
            "qty_sold": "Qty_Sold_Kg",
            "maxprice": "MaxPrice",
            "avgprice": "AvgPrice",
        })
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        for col in ["Lots", "Qty_Arrived_Kg", "Qty_Sold_Kg", "MaxPrice", "AvgPrice"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["Date", "Auctioneer", "Lots", "Qty_Arrived_Kg",
                  "Qty_Sold_Kg", "MaxPrice", "AvgPrice"]]
        df = df.dropna(subset=["Date", "AvgPrice"])

        log.info(f"Fetched {len(df)} rows from daily-price page "
                 f"({df['Date'].min().date()} to {df['Date'].max().date()})")
        return df
    except Exception as e:
        log.warning(f"Daily-price page fetch failed: {e}")
        return None


def scrape_auction_page() -> pd.DataFrame | None:
    """Scrape today's auction data from indianspices.com ticker text."""
    try:
        resp = requests.get(SPICES_BOARD_URL, timeout=30, headers={
            "User-Agent": "CardamomPulse/1.0 (price-research)"
        })
        resp.raise_for_status()

        # Strip HTML tags to get clean text for regex matching
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


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-auctioneer rows to daily totals (volume-weighted avg price)."""

    def _wavg(group):
        sold = group["Qty_Sold_Kg"]
        weights = sold / sold.sum() if sold.sum() > 0 else 1 / len(sold)
        return pd.Series({
            "Lots": group["Lots"].sum(),
            "Qty_Arrived_Kg": group["Qty_Arrived_Kg"].sum(),
            "Qty_Sold_Kg": sold.sum(),
            "AvgPrice": (group["AvgPrice"] * weights).sum(),
            "MaxPrice": group["MaxPrice"].max(),
        })

    daily = df.groupby("Date").apply(_wavg, include_groups=False).reset_index()
    daily = daily.rename(columns={
        "Date": "date",
        "Lots": "lots",
        "Qty_Arrived_Kg": "arrived_kg",
        "Qty_Sold_Kg": "sold_kg",
        "AvgPrice": "avg_price",
        "MaxPrice": "max_price",
    })
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def detect_gaps(min_gap_days: int = 3) -> list[dict]:
    """Detect gaps in auction_daily where consecutive trading days are too far apart.

    Returns list of dicts with 'gap_start', 'gap_end', 'gap_days' for each gap.
    Weekends (2 days) and normal holidays (up to min_gap_days-1) are ignored.
    """
    conn = get_conn()
    dates = pd.read_sql("SELECT date FROM auction_daily ORDER BY date", conn)
    conn.close()

    if len(dates) < 2:
        return []

    dates["date"] = pd.to_datetime(dates["date"])
    gaps = []
    for i in range(1, len(dates)):
        delta = (dates["date"].iloc[i] - dates["date"].iloc[i - 1]).days
        if delta >= min_gap_days:
            gaps.append({
                "gap_start": dates["date"].iloc[i - 1].strftime("%Y-%m-%d"),
                "gap_end": dates["date"].iloc[i].strftime("%Y-%m-%d"),
                "gap_days": delta,
            })
    return gaps


def _upsert_daily(daily: pd.DataFrame, label: str) -> int:
    """INSERT OR IGNORE aggregated daily rows into auction_daily. Returns row count."""
    if daily.empty:
        return 0
    conn = get_conn()
    cols = "date, lots, arrived_kg, sold_kg, avg_price, max_price"
    sql = f"INSERT OR IGNORE INTO auction_daily ({cols}) VALUES (?, ?, ?, ?, ?, ?)"
    conn.executemany(sql, daily.values.tolist())
    conn.commit()
    conn.close()
    log.info(f"Backfill from {label}: {len(daily)} rows offered")
    return len(daily)


def backfill_auction() -> dict:
    """Detect gaps in auction data and attempt to fill them.

    Tries sources in order of coverage:
      1. Daily-price page (full history as embedded JSON — best source)
      2. Ticker scrape (recent days only)
      3. Local XLS fallback (useful if updated manually)

    Returns summary dict with gaps found, filled, and remaining.
    """
    gaps_before = detect_gaps()
    if not gaps_before:
        log.info("No data gaps detected in auction_daily")
        return {"gaps_found": 0, "gaps_filled": 0, "gaps_remaining": []}

    log.info(f"Found {len(gaps_before)} gap(s) in auction_daily")
    for g in gaps_before:
        log.info(f"  Gap: {g['gap_start']} → {g['gap_end']} ({g['gap_days']} days)")

    # Source 1: Daily-price page — embeds full auction history as JSON
    price_data = fetch_daily_price_data()
    if price_data is not None and len(price_data) > 0:
        _upsert_daily(aggregate_daily(price_data), "daily-price page")
    else:
        log.info("Daily-price page unavailable, trying other sources")

    # Source 2: Ticker scrape — may have a few recent days
    scraped = scrape_auction_page()
    if scraped is not None and len(scraped) > 0:
        _upsert_daily(aggregate_daily(scraped), "ticker scrape")

    # Source 3: Local XLS — useful if manually refreshed
    try:
        _upsert_daily(aggregate_daily(load_xls_fallback()), "XLS fallback")
    except FileNotFoundError:
        log.warning("XLS fallback file not found")

    # Check remaining gaps
    gaps_after = detect_gaps()
    filled = len(gaps_before) - len(gaps_after)

    for g in gaps_after:
        log.warning(
            f"Unfilled gap: {g['gap_start']} → {g['gap_end']} ({g['gap_days']} days) "
            f"— market may have been closed"
        )

    return {
        "gaps_found": len(gaps_before),
        "gaps_filled": filled,
        "gaps_remaining": gaps_after,
    }


def collect_auction() -> pd.DataFrame:
    """Main entry: fetch auction data, aggregate, and upsert.

    Tries sources in order of freshness:
      1. Daily-price page (full history, updated daily by Spices Board)
      2. Ticker scrape (recent days from the auction page)
      3. Local XLS fallback (static file, may be stale)
    All available data is merged before aggregation.
    """
    frames: list[pd.DataFrame] = []

    # Best source: daily-price page with full embedded JSON
    price_data = fetch_daily_price_data()
    if price_data is not None and len(price_data) > 0:
        frames.append(price_data)
        log.info(f"Daily-price page: {len(price_data)} rows")

    # Supplement with ticker scrape (may have today's data before
    # the daily-price page is updated)
    scraped = scrape_auction_page()
    if scraped is not None and len(scraped) > 0:
        frames.append(scraped)
        log.info(f"Ticker scrape: {len(scraped)} rows")

    # Fall back to local XLS if online sources failed
    if not frames:
        log.warning("Online sources unavailable, falling back to local XLS")
    try:
        frames.append(load_xls_fallback())
    except FileNotFoundError:
        if not frames:
            raise RuntimeError("No auction data sources available")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["Date", "Auctioneer"], keep="last")
    daily = aggregate_daily(raw)

    # Upsert into DB
    conn = get_conn()
    cols = "date, lots, arrived_kg, sold_kg, avg_price, max_price"
    sql = f"INSERT OR REPLACE INTO auction_daily ({cols}) VALUES (?, ?, ?, ?, ?, ?)"
    conn.executemany(sql, daily.values.tolist())
    conn.commit()
    conn.close()

    return daily
