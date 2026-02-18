"""Collect financial data (USD/INR, Crude, Gold, Nifty) via Yahoo Finance."""

import logging
from datetime import date, timedelta
import pandas as pd
import requests
from pipeline.config import CSV_DIR, YF_TICKERS
from pipeline.db import get_conn, get_latest_date

log = logging.getLogger(__name__)


def _load_yf_csv(filename: str, value_col: str) -> pd.DataFrame:
    """Load a Yahoo Finance CSV (skip first 3 rows, extract Date+Close)."""
    path = CSV_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(path), skiprows=3)
    if "Date" not in df.columns:
        # Try reading differently
        df = pd.read_csv(str(path))
        if "Date" not in df.columns:
            return pd.DataFrame()
    df = df[["Date", "Close"]].copy()
    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna()
    return df


def _fetch_yf_api(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch from Yahoo Finance v8 chart API (no library needed)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(pd.Timestamp(start).timestamp()),
        "period2": int(pd.Timestamp(end).timestamp()),
        "interval": "1d",
    }
    headers = {"User-Agent": "CardamomPulse/1.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({
            "date": pd.to_datetime(timestamps, unit="s").strftime("%Y-%m-%d"),
            "close": closes,
        })
        return df.dropna()
    except Exception as e:
        log.warning(f"Yahoo Finance API failed for {ticker}: {e}")
        return None


def collect_finance() -> pd.DataFrame:
    """Collect all financial indicators, merge into single daily DataFrame."""
    csv_map = {
        "USDINR": ("external_usdinr.csv", "usdinr"),
        "CrudeOil": ("external_crude_oil.csv", "crude_oil"),
        "Gold": ("external_gold.csv", "gold"),
        "Nifty": ("external_nifty50.csv", "nifty"),
    }

    all_dfs = []
    for name, (filename, col_name) in csv_map.items():
        # Load CSV baseline
        csv_df = _load_yf_csv(filename, col_name)

        # Try API for recent data
        ticker = YF_TICKERS[name]
        latest = None
        if len(csv_df) > 0:
            latest = csv_df["date"].max()

        if latest:
            start = (pd.Timestamp(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = "2014-11-01"

        end = date.today().strftime("%Y-%m-%d")
        if start <= end:
            api_df = _fetch_yf_api(ticker, start, end)
            if api_df is not None and len(api_df) > 0:
                api_df = api_df.rename(columns={"close": col_name})
                csv_df = pd.concat([csv_df, api_df], ignore_index=True)
                csv_df = csv_df.drop_duplicates(subset=["date"], keep="last")
                log.info(f"Fetched {len(api_df)} new rows for {name}")

        csv_df = csv_df.sort_values("date").reset_index(drop=True)
        all_dfs.append(csv_df.set_index("date"))

    # Merge all on date
    if not all_dfs:
        return pd.DataFrame()

    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, how="outer")
    combined = combined.ffill().reset_index()
    combined = combined.rename(columns={"index": "date"})

    # Upsert into DB
    conn = get_conn()
    sql = """INSERT OR REPLACE INTO finance_daily
             (date, usdinr, crude_oil, gold, nifty)
             VALUES (?, ?, ?, ?, ?)"""
    rows = combined[["date", "usdinr", "crude_oil", "gold", "nifty"]].values.tolist()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()

    return combined
