"""Pipeline configuration constants."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "cardamom.db"
MODELS_DIR = DATA_DIR / "models"
EXPORT_DIR = ROOT_DIR / "cardamom_webapp" / "data"
CSV_DIR = ROOT_DIR  # existing external_*.csv files live at repo root

# ── Data source URLs ──────────────────────────────────────────────────────
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
IDUKKI_LAT, IDUKKI_LON = 9.85, 77.0
GUATEMALA_LAT, GUATEMALA_LON = 15.47, -90.37

ENSO_ONI_URL = (
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
)

# Yahoo Finance tickers (fetched via requests, no yfinance needed)
YF_TICKERS = {
    "USDINR": "USDINR=X",
    "CrudeOil": "BZ=F",
    "Gold": "GC=F",
    "Nifty": "^NSEI",
}

# ── Walk-forward CV config ────────────────────────────────────────────────
WF_CONFIG = {
    "daily": {"min_train": 500, "step": 120, "eval_win": 120},
    "weekly": {"min_train": 104, "step": 13, "eval_win": 13},
    "monthly": {"min_train": 48, "step": 6, "eval_win": 6},
}

# ── Model versions ────────────────────────────────────────────────────────
MODEL_VERSION = "v1.0"

# ── Pipeline schedule (IST offsets in comments) ──────────────────────────
DAILY_RUN_HOUR_UTC = 12  # 6 PM IST = 12:30 UTC; run at 12 UTC
WEEKLY_RETRAIN_DAY = 6   # Sunday
MONTHLY_FETCH_DAY = 1    # 1st of month


def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, EXPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
