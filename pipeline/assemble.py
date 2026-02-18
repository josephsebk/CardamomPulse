"""Assemble all data sources into daily/weekly/monthly DataFrames for modeling."""

import logging
import pandas as pd
from pipeline.db import get_conn, read_table

log = logging.getLogger(__name__)


def build_daily_df() -> pd.DataFrame:
    """Merge all data sources into a single daily DataFrame."""
    conn = get_conn()

    # Base: auction daily prices
    auction = read_table("auction_daily", conn)
    if auction.empty:
        conn.close()
        return pd.DataFrame()

    auction["date"] = pd.to_datetime(auction["date"])
    df = auction.sort_values("date").reset_index(drop=True)

    # Weather Idukki
    weather_id = read_table("weather_idukki", conn)
    if not weather_id.empty:
        weather_id["date"] = pd.to_datetime(weather_id["date"])
        df = pd.merge(df, weather_id, on="date", how="left")

    # Weather Guatemala
    weather_gt = read_table("weather_guatemala", conn)
    if not weather_gt.empty:
        weather_gt["date"] = pd.to_datetime(weather_gt["date"])
        df = pd.merge(df, weather_gt, on="date", how="left")

    # Finance
    finance = read_table("finance_daily", conn)
    if not finance.empty:
        finance["date"] = pd.to_datetime(finance["date"])
        df = pd.merge(df, finance, on="date", how="left")
        # Forward fill financial data (weekends/holidays)
        for col in ["usdinr", "crude_oil", "gold", "nifty"]:
            if col in df.columns:
                df[col] = df[col].ffill()

    # ENSO (stored as monthly, expand to daily via merge_asof)
    enso = read_table("enso_monthly", conn)
    if not enso.empty:
        # Reconstruct daily ENSO from monthly
        season_to_month = {
            "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
            "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
            "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
        }
        enso["month"] = enso["season"].map(season_to_month)
        enso["enso_date"] = pd.to_datetime(
            enso["year"].astype(str) + "-" + enso["month"].astype(str) + "-15",
            errors="coerce",
        )
        enso = enso.dropna(subset=["enso_date"]).sort_values("enso_date")
        enso_daily = enso.set_index("enso_date")[["anomaly"]].resample("D").ffill().reset_index()
        enso_daily.columns = ["date", "ENSO"]
        df = pd.merge_asof(df.sort_values("date"), enso_daily, on="date", direction="backward")

    # Festival calendar
    festivals = read_table("festival_calendar", conn)
    if not festivals.empty:
        festivals["date"] = pd.to_datetime(festivals["date"])
        df = pd.merge(df, festivals, on="date", how="left")

    # Trade data (monthly → merged backward into daily)
    guatemala = read_table("trade_guatemala", conn)
    if not guatemala.empty:
        guatemala["date"] = pd.to_datetime(guatemala["period"].apply(
            lambda p: pd.Timestamp(year=int(str(int(p))[:4]), month=int(str(int(p))[4:6]), day=1)
            + pd.offsets.MonthEnd(0)
        ))
        gt = guatemala[["date", "qty_kg", "net_wgt_kg", "value_usd"]].copy()
        gt["gt_qty_kg"] = gt.apply(
            lambda r: r["net_wgt_kg"] if (pd.isna(r["qty_kg"]) or r["qty_kg"] == 0) else r["qty_kg"],
            axis=1,
        )
        gt["gt_value_usd"] = gt["value_usd"]
        gt = gt[["date", "gt_qty_kg", "gt_value_usd"]].sort_values("date")
        df = pd.merge_asof(df.sort_values("date"), gt, on="date", direction="backward")

    saudi = read_table("trade_saudi", conn)
    if not saudi.empty:
        saudi["date"] = pd.to_datetime(saudi["period"].apply(
            lambda p: pd.Timestamp(year=int(str(int(p))[:4]), month=int(str(int(p))[4:6]), day=1)
            + pd.offsets.MonthEnd(0)
        ))
        sa = saudi[["date", "net_wgt_kg", "value_usd"]].copy()
        sa.columns = ["date", "sa_qty_kg", "sa_value_usd"]
        sa = sa.sort_values("date")
        df = pd.merge_asof(df.sort_values("date"), sa, on="date", direction="backward")

    # Google Trends (monthly → backward merge)
    trends = read_table("google_trends", conn)
    if not trends.empty:
        trends["date"] = pd.to_datetime(trends["date"])
        trends = trends.sort_values("date")
        df = pd.merge_asof(
            df.sort_values("date"),
            trends[["date", "cardamom_price", "cardamom_plantation"]],
            on="date", direction="backward",
        )

    conn.close()

    # Convert date back to string for consistency
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"Assembled daily DataFrame: {df.shape}")
    return df


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly (Friday close)."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Numeric columns to aggregate
    agg = {"avg_price": "last", "max_price": "last"}
    if "sold_kg" in df.columns:
        agg["sold_kg"] = "sum"
    if "arrived_kg" in df.columns:
        agg["arrived_kg"] = "sum"

    # Sum weather, forward-fill the rest
    for col in ["rain_mm", "gt_rain_mm"]:
        if col in df.columns:
            agg[col] = "sum"

    # Forward-fill columns (take last value of the week)
    ff_cols = [
        "usdinr", "crude_oil", "gold", "nifty", "ENSO",
        "gt_qty_kg", "gt_value_usd", "sa_qty_kg", "sa_value_usd",
        "temp_max_c", "humidity_pct", "cardamom_price", "cardamom_plantation",
        "wedding_season", "harvest_season", "peak_harvest",
        "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    ]
    for col in ff_cols:
        if col in df.columns:
            agg[col] = "last"

    weekly = df.resample("W-FRI").agg(agg).reset_index()
    weekly = weekly.dropna(subset=["avg_price"])
    weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
    log.info(f"Weekly DataFrame: {weekly.shape}")
    return weekly


def resample_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to month-end."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    agg = {"avg_price": "last", "max_price": "last"}
    if "sold_kg" in df.columns:
        agg["sold_kg"] = "sum"
    if "arrived_kg" in df.columns:
        agg["arrived_kg"] = "sum"

    for col in ["rain_mm", "gt_rain_mm"]:
        if col in df.columns:
            agg[col] = "sum"

    ff_cols = [
        "usdinr", "crude_oil", "gold", "nifty", "ENSO",
        "gt_qty_kg", "gt_value_usd", "sa_qty_kg", "sa_value_usd",
        "temp_max_c", "humidity_pct", "cardamom_price", "cardamom_plantation",
        "wedding_season", "harvest_season", "peak_harvest",
        "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    ]
    for col in ff_cols:
        if col in df.columns:
            agg[col] = "last"

    monthly = df.resample("ME").agg(agg).reset_index()
    monthly = monthly.dropna(subset=["avg_price"])
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")
    log.info(f"Monthly DataFrame: {monthly.shape}")
    return monthly
