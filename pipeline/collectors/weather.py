"""Collect historical weather and 14-day forward weather forecasts from Open-Meteo."""

import logging
from datetime import date, timedelta
import pandas as pd
import requests
from pipeline.config import (
    CSV_DIR,
    GUATEMALA_LAT,
    GUATEMALA_LON,
    IDUKKI_LAT,
    IDUKKI_LON,
    OPEN_METEO_BASE,
    OPEN_METEO_FORECAST_BASE,
)
from pipeline.db import get_conn, get_latest_date

log = logging.getLogger(__name__)


def _fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Fetch historical weather from Open-Meteo archive API."""
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
        return pd.DataFrame({
            "date": daily["time"],
            "precipitation_mm": daily.get("precipitation_sum"),
            "rain_mm": daily.get("rain_sum"),
            "temp_max_c": daily.get("temperature_2m_max"),
            "temp_min_c": daily.get("temperature_2m_min"),
            "humidity_pct": daily.get("relative_humidity_2m_mean"),
        })
    except Exception as e:
        log.warning(f"Open-Meteo archive fetch failed ({lat},{lon}): {e}")
        return None


def _fetch_open_meteo_forecast(lat: float, lon: float, days: int = 14) -> tuple[float | None, float | None]:
    """Fetch forward-looking 14-day rainfall sum and mean max temp from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": days,
        "daily": "rain_sum,temperature_2m_max",
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_FORECAST_BASE, params=params, timeout=30)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        rains = [float(x) for x in (daily.get("rain_sum") or []) if x is not None]
        temps = [float(x) for x in (daily.get("temperature_2m_max") or []) if x is not None]
        rain_sum = sum(rains) if rains else None
        temp_mean = (sum(temps) / len(temps)) if temps else None
        return rain_sum, temp_mean
    except Exception as e:
        log.warning(f"Open-Meteo forecast fetch failed ({lat},{lon}): {e}")
        return None, None


def _load_csv_fallback(filename: str) -> pd.DataFrame:
    """Load from existing CSV file."""
    path = CSV_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(path))
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    return df


def collect_weather_idukki() -> pd.DataFrame:
    """Collect Idukki historical weather: merge CSV history with Open-Meteo API."""
    csv_df = _load_csv_fallback("external_idukki_weather.csv")

    latest = get_latest_date("weather_idukki")
    if latest:
        start = (pd.Timestamp(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
    elif len(csv_df) > 0:
        start = csv_df["date"].max()
    else:
        start = "2014-01-01"

    end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    api_df = None
    if start <= end:
        api_df = _fetch_open_meteo(IDUKKI_LAT, IDUKKI_LON, start, end)
        if api_df is not None:
            log.info(f"Fetched {len(api_df)} Idukki weather rows from API")

    frames = []
    if len(csv_df) > 0:
        frames.append(csv_df[["date", "precipitation_mm", "rain_mm", "temp_max_c",
                              "temp_min_c", "humidity_pct"]].copy())
    if api_df is not None and len(api_df) > 0:
        frames.append(api_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

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
    """Collect Guatemala historical weather: merge CSV history with Open-Meteo API."""
    csv_df = _load_csv_fallback("external_guatemala_weather.csv")

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
            }).drop(columns=["humidity_pct"], errors="ignore")
            log.info(f"Fetched {len(api_df)} Guatemala weather rows from API")

    frames = []
    if len(csv_df) > 0:
        frames.append(csv_df[["date", "gt_precipitation_mm", "gt_rain_mm",
                              "gt_temp_max_c", "gt_temp_min_c"]].copy())
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


def collect_weather_forecast() -> dict:
    """Fetch 14-day forward-looking weather forecasts for Idukki and Guatemala.

    Stored daily in `weather_forecast` so models can use genuinely causal,
    forward-looking monsoon/rainfall stress signals without lookahead.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    id_rain, id_temp = _fetch_open_meteo_forecast(IDUKKI_LAT, IDUKKI_LON, days=14)
    gt_rain, gt_temp = _fetch_open_meteo_forecast(GUATEMALA_LAT, GUATEMALA_LON, days=14)

    if id_rain is None and gt_rain is None:
        return {}

    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO weather_forecast
           (date, idukki_fcst_rain_14d, idukki_fcst_temp_14d,
            gt_fcst_rain_14d, gt_fcst_temp_14d)
           VALUES (?, ?, ?, ?, ?)""",
        (today_str, id_rain, id_temp, gt_rain, gt_temp),
    )
    conn.commit()
    conn.close()
    log.info(
        f"Stored 14-day weather forecast ({today_str}): "
        f"Idukki rain={id_rain}mm, Guatemala rain={gt_rain}mm"
    )
    return {
        "date": today_str,
        "idukki_fcst_rain_14d": id_rain,
        "idukki_fcst_temp_14d": id_temp,
        "gt_fcst_rain_14d": gt_rain,
        "gt_fcst_temp_14d": gt_temp,
    }
