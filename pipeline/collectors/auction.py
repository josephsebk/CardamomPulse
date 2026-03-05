"""Collect daily cardamom auction prices from Spices Board / XLS fallback."""

import logging
import re
import pandas as pd
import requests
from pipeline.config import CSV_DIR
from pipeline.db import get_conn, get_latest_date

log = logging.getLogger(__name__)

SPICES_BOARD_URL = "https://www.indianspices.com/marketing/auction/small-cardamom.html"

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


def backfill_auction() -> dict:
    """Detect gaps in auction data and attempt to fill them.

    Tries two sources in order:
      1. Scrape the Spices Board page (may contain multiple recent days)
      2. Re-import the XLS fallback (covers historical data)

    Returns summary dict with gaps found, filled, and remaining.
    """
    gaps_before = detect_gaps()
    if not gaps_before:
        log.info("No data gaps detected in auction_daily")
        return {"gaps_found": 0, "gaps_filled": 0, "gaps_remaining": []}

    log.info(f"Found {len(gaps_before)} gap(s) in auction_daily")
    for g in gaps_before:
        log.info(f"  Gap: {g['gap_start']} → {g['gap_end']} ({g['gap_days']} days)")

    # Source 1: Scrape the Spices Board page — it sometimes shows
    # multiple days of ticker entries, not just today's.
    scraped = scrape_auction_page()
    if scraped is not None and len(scraped) > 0:
        daily_scraped = aggregate_daily(scraped)
        conn = get_conn()
        cols = "date, lots, arrived_kg, sold_kg, avg_price, max_price"
        placeholders = "?, ?, ?, ?, ?, ?"
        sql = f"INSERT OR IGNORE INTO auction_daily ({cols}) VALUES ({placeholders})"
        conn.executemany(sql, daily_scraped.values.tolist())
        conn.commit()
        conn.close()
        log.info(f"Scraped {len(daily_scraped)} day(s) from Spices Board for backfill")
    else:
        log.info("No data available from Spices Board scrape")

    # Source 2: Re-import XLS — useful if the file was updated externally
    # since the last collect_auction() run.
    try:
        raw = load_xls_fallback()
        daily_xls = aggregate_daily(raw)
        conn = get_conn()
        cols = "date, lots, arrived_kg, sold_kg, avg_price, max_price"
        placeholders = "?, ?, ?, ?, ?, ?"
        sql = f"INSERT OR IGNORE INTO auction_daily ({cols}) VALUES ({placeholders})"
        conn.executemany(sql, daily_xls.values.tolist())
        conn.commit()
        conn.close()
        log.info(f"Re-imported XLS fallback ({len(daily_xls)} rows)")
    except FileNotFoundError:
        log.warning("XLS fallback file not found — cannot backfill from XLS")

    # Check remaining gaps
    gaps_after = detect_gaps()
    filled = len(gaps_before) - len(gaps_after)

    for g in gaps_after:
        log.warning(
            f"Unfilled gap: {g['gap_start']} → {g['gap_end']} ({g['gap_days']} days) "
            f"— market may have been closed, or update the XLS file manually"
        )

    return {
        "gaps_found": len(gaps_before),
        "gaps_filled": filled,
        "gaps_remaining": gaps_after,
    }


def collect_auction() -> pd.DataFrame:
    """Main entry: load XLS history, merge with fresh scrape, aggregate, upsert."""
    # Always load XLS baseline for historical data
    raw = load_xls_fallback()
    log.info(f"Loaded XLS fallback: {len(raw)} rows")

    # Try to scrape today's data and merge it in
    scraped = scrape_auction_page()
    if scraped is not None and len(scraped) > 0:
        raw = pd.concat([raw, scraped], ignore_index=True)
        raw = raw.drop_duplicates(subset=["Date", "Auctioneer"], keep="last")
        log.info(f"Merged {len(scraped)} scraped rows with XLS history")

    daily = aggregate_daily(raw)

    # Upsert into DB
    conn = get_conn()
    cols = "date, lots, arrived_kg, sold_kg, avg_price, max_price"
    placeholders = "?, ?, ?, ?, ?, ?"
    sql = f"INSERT OR REPLACE INTO auction_daily ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, daily.values.tolist())
    conn.commit()
    conn.close()

    return daily
