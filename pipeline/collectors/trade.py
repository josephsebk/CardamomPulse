"""Collect trade data (UN Comtrade v1 API + CSV baseline), Google Trends, and festival calendar."""

import logging
import os
from datetime import date
import pandas as pd
import requests
from pipeline.config import (
    COMTRADE_API_BASE,
    COMTRADE_CARDAMOM_CODES,
    COMTRADE_PREVIEW_BASE,
    CSV_DIR,
)
from pipeline.db import get_conn

log = logging.getLogger(__name__)


def _period_to_date(period: int | str) -> str:
    """Convert YYYYMM period to last day of month."""
    s = str(int(period))
    year, month = int(s[:4]), int(s[4:6])
    dt = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return dt.strftime("%Y-%m-%d")


def _recent_periods(after_period: str | None) -> list[str]:
    """Build list of YYYYMM strings from the month after `after_period` up to current month."""
    today = date.today()
    if after_period and len(str(after_period)) == 6:
        y = int(str(after_period)[:4])
        m = int(str(after_period)[4:6]) + 1
        if m > 12:
            y, m = y + 1, 1
    else:
        y, m = today.year - 1, 1

    periods = []
    while (y, m) <= (today.year, today.month):
        periods.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return periods


def _fetch_comtrade_monthly(
    reporter_code: str,
    flow_code: str,
    after_period: str | None,
) -> pd.DataFrame | None:
    """Fetch fresh monthly cardamom trade flows from UN Comtrade v1 API.

    Uses the authenticated v1 endpoint when `COMTRADE_API_KEY` is configured,
    and falls back to the unauthenticated public preview endpoint otherwise.
    Aggregates HS 090831 (neither crushed nor ground) + 090832 (crushed/ground)
    per YYYYMM period with world partner (partnerCode=0).
    """
    periods = _recent_periods(after_period)
    if not periods:
        return None

    api_key = os.environ.get("COMTRADE_API_KEY", "").strip()
    base_url = COMTRADE_API_BASE if api_key else COMTRADE_PREVIEW_BASE
    headers = {"User-Agent": "CardamomPulse/3.0"}
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key

    rows = []
    # Chunk periods to stay well inside URL/API limits (max 12 periods per call)
    for idx in range(0, len(periods), 12):
        chunk = periods[idx: idx + 12]
        params = {
            "reporterCode": reporter_code,
            "period": ",".join(chunk),
            "partnerCode": "0",
            "partner2Code": "0",
            "cmdCode": COMTRADE_CARDAMOM_CODES,
            "flowCode": flow_code,
            "customsCode": "C00",
            "motCode": "0",
        }
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=30)
            if resp.status_code in (401, 403, 429):
                log.warning(
                    f"UN Comtrade API returned HTTP {resp.status_code} for "
                    f"reporter={reporter_code}; falling back to CSV baseline"
                )
                return None
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("data", []):
                period = str(item.get("period", "")).strip()
                if len(period) != 6 or not period.isdigit():
                    continue
                rows.append({
                    "period": period,
                    "value_usd": float(item.get("primaryValue") or 0.0),
                    "qty_kg": float(item.get("qty") or 0.0),
                    "netWgt_kg": float(item.get("netWgt") or 0.0),
                })
        except Exception as e:
            log.warning(f"UN Comtrade fetch failed (reporter={reporter_code}): {e}")
            return None

    if not rows:
        return None

    fetched = pd.DataFrame(rows)
    fetched = (
        fetched.groupby("period", as_index=False)[["value_usd", "qty_kg", "netWgt_kg"]]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )
    log.info(
        f"Fetched {len(fetched)} new UN Comtrade periods for "
        f"reporter={reporter_code} ({fetched['period'].min()}..{fetched['period'].max()})"
    )
    return fetched


def collect_guatemala_exports() -> pd.DataFrame:
    """Load Guatemala export CSV baseline, refresh via UN Comtrade v1 API, store in DB."""
    path = CSV_DIR / "external_guatemala_monthly_exports.csv"
    df = pd.read_csv(str(path)) if path.exists() else pd.DataFrame(
        columns=["period", "value_usd", "qty_kg", "netWgt_kg"]
    )
    if not df.empty:
        df["period"] = df["period"].astype(int).astype(str)

    latest_period = df["period"].max() if not df.empty else None
    fresh = _fetch_comtrade_monthly(reporter_code="320", flow_code="X", after_period=latest_period)
    if fresh is not None and not fresh.empty:
        df = pd.concat([df, fresh], ignore_index=True)
        df = df.drop_duplicates(subset=["period"], keep="last").sort_values("period")

    if df.empty:
        log.warning("Guatemala exports data unavailable")
        return pd.DataFrame()

    df["date"] = df["period"].apply(_period_to_date)
    df["qty_best"] = df.apply(
        lambda r: r["netWgt_kg"] if (pd.isna(r["qty_kg"]) or r["qty_kg"] == 0) else r["qty_kg"],
        axis=1,
    )

    conn = get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO trade_guatemala
           (period, value_usd, qty_kg, net_wgt_kg)
           VALUES (?, ?, ?, ?)""",
        [
            (str(row["period"]), float(row["value_usd"]),
             float(row["qty_kg"]) if pd.notna(row.get("qty_kg")) else 0.0,
             float(row["netWgt_kg"]) if pd.notna(row.get("netWgt_kg")) else 0.0)
            for _, row in df.iterrows()
        ],
    )
    conn.commit()
    conn.close()

    result = df[["date", "qty_best", "value_usd"]].copy()
    result.columns = ["date", "gt_qty_kg", "gt_value_usd"]
    result = result.sort_values("date").reset_index(drop=True)
    log.info(f"Loaded {len(result)} Guatemala export rows (latest {df['period'].max()})")
    return result


def collect_saudi_imports() -> pd.DataFrame:
    """Load Saudi import CSV baseline, refresh via UN Comtrade v1 API, store in DB."""
    path = CSV_DIR / "external_saudi_monthly_imports.csv"
    df = pd.read_csv(str(path)) if path.exists() else pd.DataFrame(
        columns=["period", "value_usd", "qty_kg", "netWgt_kg"]
    )
    if not df.empty:
        df["period"] = df["period"].astype(int).astype(str)

    latest_period = df["period"].max() if not df.empty else None
    fresh = _fetch_comtrade_monthly(reporter_code="682", flow_code="M", after_period=latest_period)
    if fresh is not None and not fresh.empty:
        df = pd.concat([df, fresh], ignore_index=True)
        df = df.drop_duplicates(subset=["period"], keep="last").sort_values("period")

    if df.empty:
        log.warning("Saudi imports data unavailable")
        return pd.DataFrame()

    df["date"] = df["period"].apply(_period_to_date)

    conn = get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO trade_saudi
           (period, value_usd, qty_kg, net_wgt_kg)
           VALUES (?, ?, ?, ?)""",
        [
            (str(row["period"]), float(row["value_usd"]),
             float(row["qty_kg"]) if pd.notna(row.get("qty_kg")) else 0.0,
             float(row["netWgt_kg"]) if pd.notna(row.get("netWgt_kg")) else 0.0)
            for _, row in df.iterrows()
        ],
    )
    conn.commit()
    conn.close()

    result = df[["date", "netWgt_kg", "value_usd"]].copy()
    result.columns = ["date", "sa_qty_kg", "sa_value_usd"]
    result = result.sort_values("date").reset_index(drop=True)
    log.info(f"Loaded {len(result)} Saudi import rows (latest {df['period'].max()})")
    return result


def collect_google_trends() -> pd.DataFrame:
    """Load Google Trends data from CSV, store in DB."""
    path = CSV_DIR / "external_google_trends_cardamom.csv"
    if not path.exists():
        log.warning("Google Trends CSV not found")
        return pd.DataFrame()

    df = pd.read_csv(str(path))
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    conn = get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO google_trends
           (date, cardamom_price, cardamom_cultivation, cardamom_farming,
            elaichi_price, cardamom_plantation)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (row["date"], row.get("cardamom_price"), row.get("cardamom_cultivation"),
             row.get("cardamom_farming"), row.get("elaichi_price"), row.get("cardamom_plantation"))
            for _, row in df.iterrows()
        ],
    )
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

    cols = ["date", "wedding_season", "harvest_season", "peak_harvest",
            "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear"]
    conn = get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO festival_calendar
           (date, wedding_season, harvest_season, peak_harvest,
            pre_eid_period, pre_diwali_period, pre_onam_period, xmas_newyear)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [tuple(row[c] for c in cols) for _, row in df.iterrows()],
    )
    conn.commit()
    conn.close()

    log.info(f"Loaded {len(df)} festival calendar rows (through {df['date'].max()})")
    return df
