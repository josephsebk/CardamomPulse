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
OPEN_METEO_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
IDUKKI_LAT, IDUKKI_LON = 9.85, 77.0
GUATEMALA_LAT, GUATEMALA_LON = 15.47, -90.37

ENSO_ONI_URL = (
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
)

# UN Comtrade v1 API (set COMTRADE_API_KEY env var for full subscription access;
# falls back to the public preview endpoint and local CSV baseline otherwise)
COMTRADE_API_BASE = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
COMTRADE_PREVIEW_BASE = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"
COMTRADE_CARDAMOM_CODES = "090831,090832"

# Yahoo Finance tickers (fetched via requests, no yfinance needed)
YF_TICKERS = {
    "USDINR": "USDINR=X",
    "CrudeOil": "BZ=F",
    "Gold": "GC=F",
    "Nifty": "^NSEI",
}

# ── Data quality & causal publication lags ────────────────────────────────
# Cap single-row sold_kg at this multiple of arrived_kg to guard against
# data-entry typos (e.g. 2021-01-02 recorded 511k kg sold vs 52k kg arrived).
AUCTION_SOLD_TO_ARRIVED_MAX_RATIO = 1.5

# Causal expanding auctioneer fixed-effect index parameters
HOUSE_EFFECT_SHRINKAGE_N = 20.0
HOUSE_EFFECT_HALFLIFE_DAYS = 180.0

# Realistic reporting delays so historical CV matches live data availability
COMTRADE_PUBLICATION_LAG_DAYS = 60
ENSO_PUBLICATION_LAG_DAYS = 45

# ── Walk-forward CV config ────────────────────────────────────────────────
WF_CONFIG = {
    "daily": {"min_train": 500, "step": 120, "eval_win": 120},
    "weekly": {"min_train": 104, "step": 13, "eval_win": 13},
    "monthly": {"min_train": 48, "step": 6, "eval_win": 6},
}

# Exponential recency weighting half-lives (in calendar days) for training
# so structural regime shifts (e.g. 2023-26 Guatemala thrips shock) are not
# overwhelmed by decade-old calm regimes.
RECENCY_HALFLIFE_DAYS = {
    "daily": 730.0,     # ~2.0 years
    "weekly": 912.5,    # ~2.5 years
    "monthly": 1095.0,  # ~3.0 years
}

# ── Feature selection ─────────────────────────────────────────────────────
# v3.0 replaces greedy full-history permutation top-k selection (which leaked
# evaluation folds into selection) with compact, frequency-specific stationary
# feature sets + strong L2/Huber regularization and nested-CV shrinkage.
FEATURE_SELECTION_K = {
    "short": None,
    "14d": None,
    "28d": None,
    "90d": None,
    "regime": None,
}

# ── Forecast sanity guard ─────────────────────────────────────────────────
MAX_SHORT_HORIZON_DEVIATION = 0.10

# ── Forecast settlement ───────────────────────────────────────────────────
TARGET_MATCH_TOLERANCE_DAYS = 7

# ── Volatility-scaled & Online Adaptive Conformal Inference (ACI) ─────────
# Target miscoverage rate (0.20 => 80% prediction interval).
# Locally-weighted conformal normalizes return residuals by trailing realized
# volatility (floored at CONFORMAL_VOL_FLOOR), and online ACI updates alpha_t
# from live validation_log outcomes with step size CONFORMAL_ACI_GAMMA.
CONFORMAL_ALPHA = 0.20
CONFORMAL_WINDOW_DAILY = 500
CONFORMAL_VOL_FLOOR = 0.008
CONFORMAL_ACI_GAMMA = 0.02
CONFORMAL_ALPHA_MIN = 0.02
CONFORMAL_ALPHA_MAX = 0.50

# ── Model versions ────────────────────────────────────────────────────────
# v2.0: regression targets switched to log-returns
# v2.1: walk-forward permutation feature selection; causal cycle features
# v2.2: auction microstructure features in daily/weekly candidate pools
# v2.3: daily targets aligned to calendar days + settlement tolerance
# v2.4: split-conformal prediction intervals on all regression horizons
# v3.0: comprehensive modeling & pipeline overhaul:
#   - Causal house-mix-adjusted daily price index + volume-weighted weekly/monthly bars
#   - Frequency-parameterized, scale-invariant stationary features (no raw INR levels)
#   - Causal t=0 auction feature alignment + realistic trade/ENSO publication lags
#   - Leak-free Pipeline(SimpleImputer, StandardScaler, model) inside CV & production
#   - Capacity right-sizing (Huber/Ridge + shallow Huber GBR), recency sample weights,
#     nested-CV shrinkage toward random walk, and term-structure coherence smoothing
#   - Volatility-scaled + online Adaptive Conformal Inference (ACI)
#   - Live UN Comtrade API + 14-day Open-Meteo weather forecast collectors
MODEL_VERSION = "v3.0"

# ── Pipeline schedule (IST offsets in comments) ──────────────────────────
DAILY_RUN_HOUR_UTC = 12  # 6 PM IST = 12:30 UTC; run at 12 UTC
WEEKLY_RETRAIN_DAY = 6   # Sunday
MONTHLY_FETCH_DAY = 1    # 1st of month


def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, EXPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
