"""Collect financial data (USD/INR, Crude, Gold, Nifty) via Yahoo Finance."""

import logging
from datetime import date, timedelta
import pandas as pd
import requests
from pipeline.config import CSV_DIR, YF_TICKERS
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def _load_yf_csv(filename: str, value_col: str) -> pd.DataFrame:
    """Load a Yahoo Finance CSV (handles both 3-row multi-index header and standard CSV)."""
    path = CSV_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=["date", value_col])

    # Standard yfinance export has header rows: Price,Close,High,... / Ticker,... / Date,,,,,
    raw = pd.read_csv(str(path))
    if "Date" in raw.columns and "Close" in raw.columns:
        df = raw[["Date", "Close"]].copy()
    elif "Price" in raw.columns and "Close" in raw.columns:
        # Multi-index CSV where row 0 is Ticker and row 1 is Date header
        df = raw.iloc[2:][["Price", "Close"]].copy()
        df.columns = ["Date", "Close"]
    else:
        raw2 = pd.read_csv(str(path), skiprows=3, header=None)
        if raw2.shape[1] < 2:
            return pd.DataFrame(columns=["date", value_col])
        df = raw2.iloc[:, :2].copy()
        df.columns = ["Date", "Close"]

    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["date"], keep="last")
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
        timestamps = result.get("timestamp") or []
        closes = result["indicators"]["quote"][0].get("close") or []
        if not timestamps or not closes:
            return None
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
        csv_df = _load_yf_csv(filename, col_name)

        ticker = YF_TICKERS[name]
        latest = csv_df["date"].max() if not csv_df.empty else None
        start = (
            (pd.Timestamp(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
            if latest else "2014-11-01"
        )
        end = date.today().strftime("%Y-%m-%d")

        if start <= end:
            api_df = _fetch_yf_api(ticker, start, end)
            if api_df is not None and len(api_df) > 0:
                api_df = api_df.rename(columns={"close": col_name})
                csv_df = pd.concat([csv_df, api_df], ignore_index=True)
                csv_df = csv_df.drop_duplicates(subset=["date"], keep="last")
                log.info(f"Fetched {len(api_df)} new rows for {name}")

        if csv_df.empty or "date" not in csv_df.columns:
            log.warning(f"No data available for {name} (CSV missing and API failed)")
            continue

        csv_df = csv_df.sort_values("date").reset_index(drop=True)
        all_dfs.append(csv_df.set_index("date"))

    if not all_dfs:
        log.warning("No financial series collected; keeping existing DB rows")
        return pd.DataFrame()

    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, how="outer")
    combined = combined.sort_index().ffill().reset_index()
    combined = combined.rename(columns={"index": "date"})

    # Ensure all expected columns exist even if one ticker failed
    for col in ["usdinr", "crude_oil", "gold", "nifty"]:
        if col not in combined.columns:
            combined[col] = None

    conn = get_conn()
    sql = """INSERT OR REPLACE INTO finance_daily
             (date, usdinr, crude_oil, gold, nifty)
             VALUES (?, ?, ?, ?, ?)"""
    rows = combined[["date", "usdinr", "crude_oil", "gold", "nifty"]].values.tolist()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()

    return combined
