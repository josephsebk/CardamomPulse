"""Collect trade data (Guatemala exports, Saudi imports) and Google Trends."""

import logging
import pandas as pd
from pipeline.config import CSV_DIR
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def _period_to_date(period: int | str) -> str:
    """Convert YYYYMM period to last day of month."""
    s = str(int(period))
    year, month = int(s[:4]), int(s[4:6])
    dt = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return dt.strftime("%Y-%m-%d")


def collect_guatemala_exports() -> pd.DataFrame:
    """Load Guatemala export data from CSV, store in DB."""
    path = CSV_DIR / "external_guatemala_monthly_exports.csv"
    if not path.exists():
        log.warning("Guatemala exports CSV not found")
        return pd.DataFrame()

    df = pd.read_csv(str(path))
    df["date"] = df["period"].apply(_period_to_date)

    # qty_best: use netWgt_kg when qty_kg is 0
    df["qty_best"] = df.apply(
        lambda r: r["netWgt_kg"] if (pd.isna(r["qty_kg"]) or r["qty_kg"] == 0) else r["qty_kg"],
        axis=1,
    )

    # Store in DB
    conn = get_conn()
    for _, row in df.iterrows():
        sql = """INSERT OR REPLACE INTO trade_guatemala
                 (period, value_usd, qty_kg, net_wgt_kg)
                 VALUES (?, ?, ?, ?)"""
        conn.execute(sql, (
            str(int(row["period"])),
            row["value_usd"],
            row.get("qty_kg"),
            row["netWgt_kg"],
        ))
    conn.commit()
    conn.close()

    result = df[["date", "qty_best", "value_usd"]].copy()
    result.columns = ["date", "gt_qty_kg", "gt_value_usd"]
    result = result.sort_values("date").reset_index(drop=True)
    log.info(f"Loaded {len(result)} Guatemala export rows")
    return result


def collect_saudi_imports() -> pd.DataFrame:
    """Load Saudi import data from CSV, store in DB."""
    path = CSV_DIR / "external_saudi_monthly_imports.csv"
    if not path.exists():
        log.warning("Saudi imports CSV not found")
        return pd.DataFrame()

    df = pd.read_csv(str(path))
    df["date"] = df["period"].apply(_period_to_date)

    conn = get_conn()
    for _, row in df.iterrows():
        sql = """INSERT OR REPLACE INTO trade_saudi
                 (period, value_usd, qty_kg, net_wgt_kg)
                 VALUES (?, ?, ?, ?)"""
        conn.execute(sql, (
            str(int(row["period"])),
            row["value_usd"],
            row.get("qty_kg"),
            row["netWgt_kg"],
        ))
    conn.commit()
    conn.close()

    result = df[["date", "netWgt_kg", "value_usd"]].copy()
    result.columns = ["date", "sa_qty_kg", "sa_value_usd"]
    result = result.sort_values("date").reset_index(drop=True)
    log.info(f"Loaded {len(result)} Saudi import rows")
    return result


def collect_google_trends() -> pd.DataFrame:
    """Load Google Trends data from CSV, store in DB."""
    path = CSV_DIR / "external_google_trends_cardamom.csv"
    if not path.exists():
        log.warning("Google Trends CSV not found")
        return pd.DataFrame()

    df = pd.read_csv(str(path))
    # Convert monthly date to end of month
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    conn = get_conn()
    for _, row in df.iterrows():
        sql = """INSERT OR REPLACE INTO google_trends
                 (date, cardamom_price, cardamom_cultivation, cardamom_farming,
                  elaichi_price, cardamom_plantation)
                 VALUES (?, ?, ?, ?, ?, ?)"""
        conn.execute(sql, (
            row["date"],
            row.get("cardamom_price"),
            row.get("cardamom_cultivation"),
            row.get("cardamom_farming"),
            row.get("elaichi_price"),
            row.get("cardamom_plantation"),
        ))
    conn.commit()
    conn.close()

    log.info(f"Loaded {len(df)} Google Trends rows")
    return df


def collect_festival_calendar() -> pd.DataFrame:
    """Load festival calendar from CSV, store in DB."""
    path = CSV_DIR / "external_festival_calendar.csv"
    if not path.exists():
        log.warning("Festival calendar CSV not found")
        return pd.DataFrame()

    df = pd.read_csv(str(path))
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    conn = get_conn()
    cols = ["date", "wedding_season", "harvest_season", "peak_harvest",
            "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear"]
    for _, row in df.iterrows():
        sql = """INSERT OR REPLACE INTO festival_calendar
                 (date, wedding_season, harvest_season, peak_harvest,
                  pre_eid_period, pre_diwali_period, pre_onam_period, xmas_newyear)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        conn.execute(sql, tuple(row[c] for c in cols))
    conn.commit()
    conn.close()

    log.info(f"Loaded {len(df)} festival calendar rows")
    return df
