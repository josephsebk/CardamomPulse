"""Collect ENSO Oceanic Nino Index data from NOAA."""

import logging
import pandas as pd
import requests
from io import StringIO
from pipeline.config import ENSO_ONI_URL, CSV_DIR
from pipeline.db import get_conn

log = logging.getLogger(__name__)

# Map NOAA season codes to month numbers (mid-month anchor)
SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def _load_csv_fallback() -> pd.DataFrame:
    """Load ENSO data from existing CSV."""
    path = CSV_DIR / "external_enso_oni.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(path))
    return df


def _fetch_noaa() -> pd.DataFrame | None:
    """Fetch ONI data from NOAA CPC."""
    try:
        resp = requests.get(ENSO_ONI_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(
            StringIO(resp.text),
            sep=r"\s+",
            names=["SEAS", "YR", "TOTAL", "ANOM"],
            skiprows=1,
        )
        df = df[df["YR"].apply(lambda x: str(x).isdigit())]
        df["YR"] = df["YR"].astype(int)
        df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
        df["ANOM"] = pd.to_numeric(df["ANOM"], errors="coerce")
        log.info(f"Fetched {len(df)} ENSO rows from NOAA")
        return df
    except Exception as e:
        log.warning(f"NOAA ENSO fetch failed: {e}")
        return None


def collect_enso() -> pd.DataFrame:
    """Collect ENSO data, store in DB, return as daily-interpolated DataFrame."""
    # Try NOAA first, fallback to CSV
    df = _fetch_noaa()
    if df is None or len(df) == 0:
        df = _load_csv_fallback()
        log.info(f"Using CSV fallback for ENSO: {len(df)} rows")

    if len(df) == 0:
        return pd.DataFrame()

    # Store raw monthly data
    conn = get_conn()
    for _, row in df.iterrows():
        sql = """INSERT OR REPLACE INTO enso_monthly
                 (year_season, year, season, total, anomaly)
                 VALUES (?, ?, ?, ?, ?)"""
        conn.execute(sql, (
            f"{int(row['YR'])}_{row['SEAS']}",
            int(row["YR"]),
            row["SEAS"],
            row["TOTAL"],
            row["ANOM"],
        ))
    conn.commit()
    conn.close()

    # Convert to daily (mid-month anchor, forward fill)
    df["month"] = df["SEAS"].map(SEASON_TO_MONTH)
    df["date"] = pd.to_datetime(
        df["YR"].astype(str) + "-" + df["month"].astype(str) + "-15",
        errors="coerce",
    )
    df = df.dropna(subset=["date"])
    daily = (
        df.set_index("date")[["ANOM"]]
        .rename(columns={"ANOM": "ENSO"})
        .resample("D")
        .ffill()
        .reset_index()
    )
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily
