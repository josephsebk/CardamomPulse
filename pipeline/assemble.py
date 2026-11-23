"""Assemble all data sources into daily/weekly/monthly DataFrames for modeling.

v3.0 improvements:
- Realistic causal publication lags on monthly UN Comtrade trade flows (+60d)
  and NOAA ONI ENSO (+45d) so historical CV matches live data availability.
- Pre-computes native monthly Tier-5 trade ratios (`3m`, `yoy`, `unit_price`)
  BEFORE merging onto daily/weekly frames so YoY is always genuinely 12 months.
- Volume-weighted weekly & monthly average prices (for both raw `avg_price` and
  causal house-adjusted `adj_price`) instead of noisy single-day "last" bars.
- Merges forward-looking 14-day Open-Meteo weather forecasts (`weather_forecast`).
"""

import logging
import numpy as np
import pandas as pd
from pipeline.config import (
    COMTRADE_PUBLICATION_LAG_DAYS,
    ENSO_PUBLICATION_LAG_DAYS,
)
from pipeline.db import get_conn, read_table

log = logging.getLogger(__name__)


def _prepare_monthly_trade(
    df: pd.DataFrame,
    qty_col: str,
    prefix: str,
    lag_days: int = COMTRADE_PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Compute native monthly trade indicators and apply publication lag."""
    if df.empty:
        return pd.DataFrame()

    t = df.copy()
    t["period_date"] = pd.to_datetime(t["period"].apply(
        lambda p: pd.Timestamp(year=int(str(int(p))[:4]), month=int(str(int(p))[4:6]), day=1)
        + pd.offsets.MonthEnd(0)
    ))
    t = t.sort_values("period_date").drop_duplicates(subset=["period_date"], keep="last")

    # Reindex to a regular monthly grid so rolling(3) and pct_change(12) are exact calendar months
    t = t.set_index("period_date").asfreq("ME")
    if qty_col == "best":
        qty = t["qty_kg"].where(t["qty_kg"].notna() & (t["qty_kg"] > 0), t["net_wgt_kg"])
    else:
        qty = t[qty_col]
    qty = pd.to_numeric(qty, errors="coerce")
    val = pd.to_numeric(t["value_usd"], errors="coerce")

    qty_mt = qty / 1000.0
    out = pd.DataFrame(index=t.index)
    out[f"{prefix}_qty_kg"] = qty
    out[f"{prefix}_value_usd"] = val
    out[f"{prefix}_qty_3m"] = qty_mt.rolling(3, min_periods=1).sum()
    out[f"{prefix}_qty_yoy"] = (qty_mt / qty_mt.shift(12).clip(lower=1e-3) - 1.0) * 100.0
    out[f"{prefix}_qty_yoy"] = out[f"{prefix}_qty_yoy"].clip(-95.0, 300.0)
    if prefix == "gt":
        out["gt_unit_price"] = val / qty.clip(lower=1.0)

    out = out.dropna(how="all").reset_index()
    # Shift by publication lag so historical rows only see trade stats once published
    out["date"] = out["period_date"] + pd.Timedelta(days=lag_days)
    return out.drop(columns=["period_date"]).sort_values("date")


def build_daily_df() -> pd.DataFrame:
    """Merge all data sources into a single daily DataFrame."""
    conn = get_conn()

    auction = read_table("auction_daily", conn)
    if auction.empty:
        conn.close()
        return pd.DataFrame()

    auction["date"] = pd.to_datetime(auction["date"])
    df = auction.sort_values("date").reset_index(drop=True)

    # Guarantee causal house-adjusted columns exist even on pre-v3.0 rows
    if "adj_price" not in df.columns:
        df["adj_price"] = df["avg_price"]
    else:
        df["adj_price"] = df["adj_price"].fillna(df["avg_price"])
    if "house_effect_log" not in df.columns:
        df["house_effect_log"] = 0.0
    else:
        df["house_effect_log"] = df["house_effect_log"].fillna(0.0)
    if "grade_spread_pct" not in df.columns and "max_price" in df.columns:
        df["grade_spread_pct"] = df["max_price"] / df["avg_price"].clip(lower=1.0) - 1.0

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

    # Forward-looking 14-day weather forecast (causal daily snapshots)
    try:
        wf = read_table("weather_forecast", conn)
    except Exception:
        wf = pd.DataFrame()
    if not wf.empty:
        wf["date"] = pd.to_datetime(wf["date"])
        df = pd.merge(df, wf, on="date", how="left")

    # Finance
    finance = read_table("finance_daily", conn)
    if not finance.empty:
        finance["date"] = pd.to_datetime(finance["date"])
        df = pd.merge(df, finance, on="date", how="left")
        for col in ["usdinr", "crude_oil", "gold", "nifty"]:
            if col in df.columns:
                df[col] = df[col].ffill()

    # ENSO ONI (3-month centered average; published ~45 days after mid-month)
    enso = read_table("enso_monthly", conn)
    if not enso.empty:
        season_to_month = {
            "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
            "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
            "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
        }
        enso["month"] = enso["season"].map(season_to_month)
        enso["enso_date"] = pd.to_datetime(
            enso["year"].astype(str) + "-" + enso["month"].astype(str) + "-15",
            errors="coerce",
        ) + pd.Timedelta(days=ENSO_PUBLICATION_LAG_DAYS)
        enso = enso.dropna(subset=["enso_date"]).sort_values("enso_date")
        enso_daily = (
            enso.drop_duplicates(subset=["enso_date"], keep="last")
            .set_index("enso_date")[["anomaly"]]
            .resample("D")
            .ffill()
            .reset_index()
        )
        enso_daily.columns = ["date", "ENSO"]
        df = pd.merge_asof(df.sort_values("date"), enso_daily, on="date", direction="backward")

    # Festival calendar
    festivals = read_table("festival_calendar", conn)
    if not festivals.empty:
        festivals["date"] = pd.to_datetime(festivals["date"])
        df = pd.merge(df, festivals, on="date", how="left")

    # Trade data (native monthly indicators + 60-day reporting lag)
    guatemala = read_table("trade_guatemala", conn)
    gt_monthly = _prepare_monthly_trade(guatemala, qty_col="best", prefix="gt")
    if not gt_monthly.empty:
        df = pd.merge_asof(df.sort_values("date"), gt_monthly, on="date", direction="backward")

    saudi = read_table("trade_saudi", conn)
    sa_monthly = _prepare_monthly_trade(saudi, qty_col="net_wgt_kg", prefix="sa")
    if not sa_monthly.empty:
        df = pd.merge_asof(df.sort_values("date"), sa_monthly, on="date", direction="backward")

    # Google Trends (monthly → backward merge)
    trends = read_table("google_trends", conn)
    if not trends.empty:
        trends["date"] = pd.to_datetime(trends["date"])
        trends = trends.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df = pd.merge_asof(
            df.sort_values("date"),
            trends[["date", "cardamom_price", "cardamom_plantation"]],
            on="date", direction="backward",
        )

    conn.close()

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"Assembled daily DataFrame: {df.shape}")
    return df


def _resample_period(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily data to calendar-aligned bars with volume-weighted prices.

    Using volume-weighted average prices for weekly/monthly bars removes
    single-session auctioneer rotation noise from the 28d and 90d targets,
    while preserving `last_price` / `last_adj_price` for reference.
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    vol = df["sold_kg"].fillna(0.0).clip(lower=0.0) if "sold_kg" in df.columns else pd.Series(1.0, index=df.index)
    df["_pv_raw"] = df["avg_price"] * vol
    adj_col = "adj_price" if "adj_price" in df.columns else "avg_price"
    df["_pv_adj"] = df[adj_col] * vol
    df["_vol_w"] = vol.where(df["avg_price"].notna(), 0.0)

    agg: dict[str, str] = {
        "avg_price": "mean",       # fallback when volume is 0
        "_pv_raw": "sum",
        "_pv_adj": "sum",
        "_vol_w": "sum",
        "max_price": "max",
    }
    if "adj_price" in df.columns:
        agg["adj_price"] = "mean"
    if "sold_kg" in df.columns:
        agg["sold_kg"] = "sum"
    if "arrived_kg" in df.columns:
        agg["arrived_kg"] = "sum"

    for col in ["rain_mm", "gt_rain_mm"]:
        if col in df.columns:
            agg[col] = "sum"

    for col in [
        "n_auctions", "price_disp_cv", "price_range_pct",
        "unsold_pct", "avg_lot_kg", "house_effect_log", "grade_spread_pct",
    ]:
        if col in df.columns:
            agg[col] = "mean"

    ff_cols = [
        "usdinr", "crude_oil", "gold", "nifty", "ENSO",
        "gt_qty_kg", "gt_value_usd", "gt_qty_3m", "gt_qty_yoy", "gt_unit_price",
        "sa_qty_kg", "sa_value_usd", "sa_qty_3m", "sa_qty_yoy",
        "temp_max_c", "humidity_pct",
        "idukki_fcst_rain_14d", "idukki_fcst_temp_14d",
        "gt_fcst_rain_14d", "gt_fcst_temp_14d",
        "cardamom_price", "cardamom_plantation",
        "wedding_season", "harvest_season", "peak_harvest",
        "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    ]
    for col in ff_cols:
        if col in df.columns:
            agg[col] = "last"

    res = df.resample(rule).agg(agg)
    # Also capture period-end close for anchoring / inspection
    res["last_price"] = df["avg_price"].resample(rule).last()
    res["last_adj_price"] = df[adj_col].resample(rule).last()

    # Replace simple mean with volume-weighted average where volume > 0
    has_vol = res["_vol_w"] > 0
    res.loc[has_vol, "avg_price"] = res.loc[has_vol, "_pv_raw"] / res.loc[has_vol, "_vol_w"]
    if "adj_price" in res.columns:
        res.loc[has_vol, "adj_price"] = res.loc[has_vol, "_pv_adj"] / res.loc[has_vol, "_vol_w"]
    else:
        res["adj_price"] = res["avg_price"]

    res = res.drop(columns=["_pv_raw", "_pv_adj", "_vol_w"])
    res = res.reset_index().dropna(subset=["avg_price"])
    res["date"] = res["date"].dt.strftime("%Y-%m-%d")
    return res


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly (Friday end, volume-weighted avg price)."""
    weekly = _resample_period(daily, "W-FRI")
    log.info(f"Weekly DataFrame (volume-weighted): {weekly.shape}")
    return weekly


def resample_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to month-end (volume-weighted avg price)."""
    monthly = _resample_period(daily, "ME")
    log.info(f"Monthly DataFrame (volume-weighted): {monthly.shape}")
    return monthly
