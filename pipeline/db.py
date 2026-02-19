"""SQLite database schema and operations."""

import sqlite3
import pandas as pd
from pipeline.config import DB_PATH, ensure_dirs


def get_conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS auction_daily (
        date        TEXT PRIMARY KEY,
        lots        REAL,
        arrived_kg  REAL,
        sold_kg     REAL,
        avg_price   REAL,
        max_price   REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_idukki (
        date            TEXT PRIMARY KEY,
        precipitation_mm REAL,
        rain_mm         REAL,
        temp_max_c      REAL,
        temp_min_c      REAL,
        humidity_pct    REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_guatemala (
        date                TEXT PRIMARY KEY,
        gt_precipitation_mm REAL,
        gt_rain_mm          REAL,
        gt_temp_max_c       REAL,
        gt_temp_min_c       REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS finance_daily (
        date      TEXT PRIMARY KEY,
        usdinr    REAL,
        crude_oil REAL,
        gold      REAL,
        nifty     REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS enso_monthly (
        year_season TEXT PRIMARY KEY,
        year        INTEGER,
        season      TEXT,
        total       REAL,
        anomaly     REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_guatemala (
        period      TEXT PRIMARY KEY,
        value_usd   REAL,
        qty_kg      REAL,
        net_wgt_kg  REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_saudi (
        period      TEXT PRIMARY KEY,
        value_usd   REAL,
        qty_kg      REAL,
        net_wgt_kg  REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS google_trends (
        date                  TEXT PRIMARY KEY,
        cardamom_price        REAL,
        cardamom_cultivation  REAL,
        cardamom_farming      REAL,
        elaichi_price         REAL,
        cardamom_plantation   REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS festival_calendar (
        date            TEXT PRIMARY KEY,
        wedding_season  INTEGER,
        harvest_season  INTEGER,
        peak_harvest    INTEGER,
        pre_eid_period  INTEGER,
        pre_diwali_period INTEGER,
        pre_onam_period INTEGER,
        xmas_newyear    INTEGER
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS forecast_ledger (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_date   TEXT NOT NULL,
        target_date     TEXT NOT NULL,
        horizon_days    INTEGER NOT NULL,
        predicted_price REAL NOT NULL,
        lower_bound     REAL,
        upper_bound     REAL,
        model_version   TEXT,
        created_at      TEXT DEFAULT (datetime('now')),
        UNIQUE(forecast_date, target_date, horizon_days)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS regime_ledger (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_date   TEXT NOT NULL UNIQUE,
        bear_probability REAL NOT NULL,
        regime_label    TEXT,
        model_version   TEXT,
        created_at      TEXT DEFAULT (datetime('now'))
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS validation_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT NOT NULL,
        horizon_days    INTEGER NOT NULL,
        predicted_price REAL,
        actual_price    REAL,
        abs_error       REAL,
        pct_error       REAL,
        created_at      TEXT DEFAULT (datetime('now')),
        UNIQUE(date, horizon_days)
    )""")

    conn.commit()
    conn.close()


def upsert_df(table: str, df: pd.DataFrame, conn: sqlite3.Connection | None = None):
    """Insert or replace a DataFrame into a table."""
    close = False
    if conn is None:
        conn = get_conn()
        close = True
    df.to_sql(table, conn, if_exists="replace", index=False)
    if close:
        conn.close()


def append_df(table: str, df: pd.DataFrame, conn: sqlite3.Connection | None = None):
    """Append rows, ignoring conflicts on primary key."""
    close = False
    if conn is None:
        conn = get_conn()
        close = True
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    sql = f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, df.values.tolist())
    conn.commit()
    if close:
        conn.close()


def read_table(table: str, conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Read entire table into DataFrame."""
    close = False
    if conn is None:
        conn = get_conn()
        close = True
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    if close:
        conn.close()
    return df


def get_latest_date(table: str, date_col: str = "date") -> str | None:
    """Get the most recent date in a table."""
    conn = get_conn()
    cur = conn.execute(f"SELECT MAX({date_col}) FROM {table}")
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None
