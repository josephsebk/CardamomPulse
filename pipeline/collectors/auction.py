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
