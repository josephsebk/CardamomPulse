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

# ── Feature selection ─────────────────────────────────────────────────────
# Top-k features kept per horizon, ranked by walk-forward permutation
# importance over the full candidate pool at retrain time.
# None = keep the hand-curated set (selection underperformed it in backtests).
FEATURE_SELECTION_K = {
    "short": 15,   # 1d-7d models (shared feature set, ranked on 7d target)
    "14d": 10,
    "28d": None,   # manual T1-T5 set beat all selected subsets
    "90d": 5,
    "regime": 20,
}

# ── Forecast sanity guard ─────────────────────────────────────────────────
# The shortest-horizon forecast must stay within this fraction of the latest
# spot price, otherwise the run aborts before archiving/publishing. A 1-day
# move of this size essentially never occurs in this market (1d CV MAPE is
# ~2.6%), so a breach signals a bug — e.g. a forecast anchored on a stale
# price row — not a real prediction.
MAX_SHORT_HORIZON_DEVIATION = 0.10

# Maximum age (calendar days) of the anchor row before predictions are
# considered unreliable. When the daily scraper fails and falls back to the
# XLS baseline (which can be months old) every prediction is silently
# anchored to a stale price level — producing large systematic errors
# without triggering MAX_SHORT_HORIZON_DEVIATION (which compares the
# forecast against the stale anchor, not the real current price).
# Short-horizon models (≤7d) are disabled outright; longer horizons get a
# staleness warning flag that the webapp can surface to users.
MAX_ANCHOR_AGE_DAYS = 5

# ── Model versions ────────────────────────────────────────────────────────
# v2.0: regression targets switched to log-returns
# v2.1: walk-forward permutation feature selection; causal cycle features
# v2.2: auction microstructure features (cross-auction dispersion, unsold
#       share, auction count, lot size) in daily/weekly candidate pools
MODEL_VERSION = "v2.2"

# ── Pipeline schedule (IST offsets in comments) ──────────────────────────
DAILY_RUN_HOUR_UTC = 12  # 6 PM IST = 12:30 UTC; run at 12 UTC
WEEKLY_RETRAIN_DAY = 6   # Sunday
MONTHLY_FETCH_DAY = 1    # 1st of month


def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, EXPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
