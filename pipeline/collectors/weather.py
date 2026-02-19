"""Collect weather data from Open-Meteo API for Idukki and Guatemala."""

import logging
from datetime import date, timedelta
import pandas as pd
import requests
from pipeline.config import (
    OPEN_METEO_BASE, IDUKKI_LAT, IDUKKI_LON,
    GUATEMALA_LAT, GUATEMALA_LON, CSV_DIR,
)
from pipeline.db import get_conn, get_latest_date

log = logging.getLogger(__name__)


def _fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Fetch weather from Open-Meteo archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean",
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        if not daily or not daily.get("time"):
            return None
        df = pd.DataFrame({
            "date": daily["time"],
            "precipitation_mm": daily.get("precipitation_sum"),
            "rain_mm": daily.get("rain_sum"),
            "temp_max_c": daily.get("temperature_2m_max"),
            "temp_min_c": daily.get("temperature_2m_min"),
            "humidity_pct": daily.get("relative_humidity_2m_mean"),
        })
        return df
    except Exception as e:
        log.warning(f"Open-Meteo fetch failed ({lat},{lon}): {e}")
        return None


def _load_csv_fallback(filename: str, col_map: dict) -> pd.DataFrame:
    """Load from existing CSV file."""
    path = CSV_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(path))
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    return df


def collect_weather_idukki() -> pd.DataFrame:
    """Collect Idukki weather: try API for recent data, merge with CSV history."""
    # Load existing CSV as baseline
    csv_df = _load_csv_fallback("external_idukki_weather.csv", {})
    if "date" not in csv_df.columns and "Date" in csv_df.columns:
        csv_df = csv_df.rename(columns={"Date": "date"})

    # Determine start date for API fetch
    latest = get_latest_date("weather_idukki")
    if latest:
        start = (pd.Timestamp(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
    elif len(csv_df) > 0:
        start = csv_df["date"].max()
    else:
        start = "2014-01-01"

    end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch new data from API
    api_df = None
    if start <= end:
        api_df = _fetch_open_meteo(IDUKKI_LAT, IDUKKI_LON, start, end)
        if api_df is not None:
            log.info(f"Fetched {len(api_df)} Idukki weather rows from API")

    # Combine CSV + API
    frames = []
    if len(csv_df) > 0:
        csv_df = csv_df.rename(columns={
            "Date": "date",
        })
        csv_std = csv_df[["date", "precipitation_mm", "rain_mm", "temp_max_c",
                          "temp_min_c", "humidity_pct"]].copy()
        frames.append(csv_std)
    if api_df is not None and len(api_df) > 0:
        frames.append(api_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    # Upsert into DB
    conn = get_conn()
    sql = """INSERT OR REPLACE INTO weather_idukki
             (date, precipitation_mm, rain_mm, temp_max_c, temp_min_c, humidity_pct)
             VALUES (?, ?, ?, ?, ?, ?)"""
    rows = combined[["date", "precipitation_mm", "rain_mm", "temp_max_c",
                     "temp_min_c", "humidity_pct"]].values.tolist()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()
    return combined


def collect_weather_guatemala() -> pd.DataFrame:
    """Collect Guatemala weather: try API for recent data, merge with CSV history."""
    csv_df = _load_csv_fallback("external_guatemala_weather.csv", {})
    if "date" not in csv_df.columns and "Date" in csv_df.columns:
        csv_df = csv_df.rename(columns={"Date": "date"})

    latest = get_latest_date("weather_guatemala")
    if latest:
        start = (pd.Timestamp(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
    elif len(csv_df) > 0:
        start = csv_df["date"].max()
    else:
        start = "2014-01-01"

    end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    api_df = None
    if start <= end:
        raw = _fetch_open_meteo(GUATEMALA_LAT, GUATEMALA_LON, start, end)
        if raw is not None:
            api_df = raw.rename(columns={
                "precipitation_mm": "gt_precipitation_mm",
                "rain_mm": "gt_rain_mm",
                "temp_max_c": "gt_temp_max_c",
                "temp_min_c": "gt_temp_min_c",
            })
            api_df = api_df.drop(columns=["humidity_pct"], errors="ignore")
            log.info(f"Fetched {len(api_df)} Guatemala weather rows from API")

    frames = []
    if len(csv_df) > 0:
        csv_std = csv_df[["date", "gt_precipitation_mm", "gt_rain_mm",
                          "gt_temp_max_c", "gt_temp_min_c"]].copy()
        frames.append(csv_std)
    if api_df is not None and len(api_df) > 0:
        frames.append(api_df[["date", "gt_precipitation_mm", "gt_rain_mm",
                              "gt_temp_max_c", "gt_temp_min_c"]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    conn = get_conn()
    sql = """INSERT OR REPLACE INTO weather_guatemala
             (date, gt_precipitation_mm, gt_rain_mm, gt_temp_max_c, gt_temp_min_c)
             VALUES (?, ?, ?, ?, ?)"""
    rows = combined[["date", "gt_precipitation_mm", "gt_rain_mm",
                     "gt_temp_max_c", "gt_temp_min_c"]].values.tolist()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()
    return combined
