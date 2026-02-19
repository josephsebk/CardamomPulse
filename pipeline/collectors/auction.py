"""Collect daily cardamom auction prices from Spices Board / XLS fallback."""

import logging
import pandas as pd
import requests
from io import StringIO
from pipeline.config import CSV_DIR
from pipeline.db import get_conn, get_latest_date

log = logging.getLogger(__name__)

SPICES_BOARD_URL = "https://www.indianspices.com/marketing/auction/small-cardamom.html"


def scrape_auction_page() -> pd.DataFrame | None:
    """Attempt to scrape today's auction data from indianspices.com."""
    try:
        resp = requests.get(SPICES_BOARD_URL, timeout=30, headers={
            "User-Agent": "CardamomPulse/1.0 (price-research)"
        })
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            log.warning("No tables found on Spices Board page")
            return None
        df = tables[0]
        log.info(f"Scraped {len(df)} rows from Spices Board")
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
    """Aggregate per-auctioneer rows to daily totals."""
    daily = (
        df.groupby("Date")
        .agg({
            "Lots": "sum",
            "Qty_Arrived_Kg": "sum",
            "Qty_Sold_Kg": "sum",
            "AvgPrice": "mean",
            "MaxPrice": "mean",
        })
        .reset_index()
    )
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
    """Main entry: try scrape, fall back to XLS, aggregate to daily, upsert."""
    scraped = scrape_auction_page()
    if scraped is not None and len(scraped) > 0:
        try:
            scraped.columns = [
                "Date", "Auctioneer", "Lots", "Qty_Arrived_Kg",
                "Qty_Sold_Kg", "MaxPrice", "AvgPrice",
            ]
            scraped["Date"] = pd.to_datetime(scraped["Date"], format="mixed", dayfirst=False)
            for col in ["Lots", "Qty_Arrived_Kg", "Qty_Sold_Kg", "MaxPrice", "AvgPrice"]:
                scraped[col] = pd.to_numeric(scraped[col], errors="coerce")
            scraped = scraped.dropna(subset=["Date", "AvgPrice"])
            daily = aggregate_daily(scraped)
            log.info(f"Scraped auction data: {len(daily)} daily rows")
        except Exception as e:
            log.warning(f"Failed to parse scraped data: {e}, falling back to XLS")
            raw = load_xls_fallback()
            daily = aggregate_daily(raw)
    else:
        raw = load_xls_fallback()
        daily = aggregate_daily(raw)
        log.info(f"Loaded XLS fallback: {len(daily)} daily rows")

    # Upsert into DB
    conn = get_conn()
    cols = "date, lots, arrived_kg, sold_kg, avg_price, max_price"
    placeholders = "?, ?, ?, ?, ?, ?"
    sql = f"INSERT OR REPLACE INTO auction_daily ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, daily.values.tolist())
    conn.commit()
    conn.close()

    return daily
