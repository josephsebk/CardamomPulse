"""Feature engineering (v3.0): frequency-aware, scale-invariant, causal features.

Key v3.0 fixes over v2.x:
1. Frequency-parameterized rolling windows (`daily`, `weekly`, `monthly`) so
   weekly/monthly frames never apply daily row counts (e.g. 182 weeks/months).
2. Purely stationary, dimensionless Tier-1 & Tier-4 features (no raw INR price
   levels or raw USD/INR levels that cause tree models to memorize old price
   regimes and systematically under-predict during supply-shock bull runs).
3. Causal `t=0` alignment for auction & microstructure features: today's auction
   has closed by the 6 PM IST run and forecasts anchor on `price[t]`, so models
   now observe today's return, MA deviation, and house-mix effect.
4. Uses the causal house-mix-adjusted index (`adj_price`) for returns/MAs while
   exposing `house_effect_log` so short-horizon models can separate market moves
   from weekday auctioneer rotation.
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=max(2, window // 2)).mean()
    avg_loss = loss.rolling(window, min_periods=max(2, window // 2)).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


# Frequency-specific window presets (in row units of each frame)
_FREQ_WINDOWS = {
    "daily": {
        "ret": [1, 2, 5, 10, 20],
        "ma": [5, 10, 20, 60],
        "rvol_short": 10,
        "rvol_long": 20,
        "rsi": 14,
        "vol_ma": 30,
        "sell_through": 14,
        "micro_ma": 14,
        "lot_base": 60,
        "rain": [28, 56, 91, 182],
        "gt_rain": [28, 91, 182],
        "rain_anom_win": 28,
        "rain_ann_win": 365,
        "temp_ma": 28,
        "enso_6m": 182,
        "enso_12m": 365,
        "macro_short": 28,
        "macro_long": 90,
        "vol_short": 60,
        "vol_2yr": 500,
    },
    "weekly": {
        "ret": [1, 2, 4, 8, 13],
        "ma": [4, 8, 13, 26],
        "rvol_short": 4,
        "rvol_long": 13,
        "rsi": 8,
        "vol_ma": 13,
        "sell_through": 4,
        "micro_ma": 4,
        "lot_base": 26,
        "rain": [4, 8, 13, 26],
        "gt_rain": [4, 13, 26],
        "rain_anom_win": 4,
        "rain_ann_win": 52,
        "temp_ma": 4,
        "enso_6m": 26,
        "enso_12m": 52,
        "macro_short": 4,
        "macro_long": 13,
        "vol_short": 13,
        "vol_2yr": 104,
    },
    "monthly": {
        "ret": [1, 2, 3, 6, 12],
        "ma": [3, 6, 12, 24],
        "rvol_short": 3,
        "rvol_long": 6,
        "rsi": 6,
        "vol_ma": 6,
        "sell_through": 3,
        "micro_ma": 3,
        "lot_base": 12,
        "rain": [1, 2, 3, 6],
        "gt_rain": [1, 3, 6],
        "rain_anom_win": 3,
        "rain_ann_win": 12,
        "temp_ma": 3,
        "enso_6m": 6,
        "enso_12m": 12,
        "macro_short": 1,
        "macro_long": 3,
        "vol_short": 3,
        "vol_2yr": 24,
    },
}


# ── Tier 1: Stationary Technical / Mean-Reversion / Volatility ───────────
# Normalized feature names shared across frequencies (windows scale with freq).

T1_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5",
    "dev_ma_1", "dev_ma_2", "dev_ma_3", "dev_ma_4",
    "rvol_short", "rvol_long", "rvol_ratio", "zscore_ma3",
    "range_pos", "rsi", "boll_pos",
    "vol_ratio", "sell_through", "max_avg_spread",
    "house_effect_log",
]


def add_tier1(
    df: pd.DataFrame,
    freq: str = "daily",
    price_col: str | None = None,
    vol_col: str = "sold_kg",
    arr_col: str = "arrived_kg",
    max_col: str = "max_price",
) -> list[str]:
    """Add stationary, scale-invariant T1 features aligned causally to row `t`."""
    cfg = _FREQ_WINDOWS[freq]
    if price_col is None:
        price_col = "adj_price" if "adj_price" in df.columns else "avg_price"

    price = pd.to_numeric(df[price_col], errors="coerce").clip(lower=1.0)
    lp = np.log(price)
    added = []

    # 1. Multi-scale log-returns (causal at t: log(P_t / P_{t-k}))
    for idx, k in enumerate(cfg["ret"], start=1):
        col = f"ret_{idx}"
        df[col] = lp.diff(k)
        added.append(col)

    # 2. Log deviations from trailing moving averages (mean-reversion terms)
    ma_cols = {}
    for idx, w in enumerate(cfg["ma"], start=1):
        col = f"dev_ma_{idx}"
        ma = lp.rolling(w, min_periods=max(2, w // 2)).mean()
        df[col] = lp - ma
        ma_cols[idx] = ma
        added.append(col)

    # 3. Realized log-return volatility (dimensionless, used by models & conformal)
    r1 = lp.diff(1)
    rs_w, rl_w = cfg["rvol_short"], cfg["rvol_long"]
    df["rvol_short"] = r1.rolling(rs_w, min_periods=max(3, rs_w // 2)).std()
    df["rvol_long"] = r1.rolling(rl_w, min_periods=max(4, rl_w // 2)).std()
    df["rvol_ratio"] = df["rvol_short"] / df["rvol_long"].clip(lower=1e-4)
    df["zscore_ma3"] = df["dev_ma_3"] / df["rvol_long"].clip(lower=1e-4)
    added.extend(["rvol_short", "rvol_long", "rvol_ratio", "zscore_ma3"])

    # 4. Bounded range position in [0, 1] over the 3rd MA window
    rng_w = cfg["ma"][2]
    rmin = lp.rolling(rng_w, min_periods=max(3, rng_w // 2)).min()
    rmax = lp.rolling(rng_w, min_periods=max(3, rng_w // 2)).max()
    df["range_pos"] = (lp - rmin) / (rmax - rmin).clip(lower=1e-4)
    added.append("range_pos")

    # 5. RSI (normalized to [-1, 1]) and Bollinger position on log-price
    df["rsi"] = (compute_rsi(lp, cfg["rsi"]) - 50.0) / 50.0
    std_ma3 = lp.rolling(rng_w, min_periods=max(3, rng_w // 2)).std()
    df["boll_pos"] = (lp - ma_cols[3]) / (2.0 * std_ma3.clip(lower=1e-4))
    added.extend(["rsi", "boll_pos"])

    # 6. Stationary volume ratio & sell-through
    if vol_col in df.columns:
        volume = pd.to_numeric(df[vol_col], errors="coerce").clip(lower=0.0)
        vol_ma = volume.rolling(cfg["vol_ma"], min_periods=max(2, cfg["vol_ma"] // 2)).mean()
        df["vol_ratio"] = np.log((volume + 1.0) / (vol_ma + 1.0))
    else:
        df["vol_ratio"] = np.nan
    added.append("vol_ratio")

    if arr_col in df.columns and vol_col in df.columns:
        arrived = pd.to_numeric(df[arr_col], errors="coerce").clip(lower=0.0)
        volume = pd.to_numeric(df[vol_col], errors="coerce").clip(lower=0.0)
        st_w = cfg["sell_through"]
        df["sell_through"] = (
            volume.rolling(st_w, min_periods=max(1, st_w // 2)).sum()
            / arrived.rolling(st_w, min_periods=max(1, st_w // 2)).sum().clip(lower=1.0)
        ).clip(0.0, 1.5)
    else:
        df["sell_through"] = np.nan
    added.append("sell_through")

    # 7. Grade/quality spread proxy (max vs avg price)
    raw_price = pd.to_numeric(df["avg_price"], errors="coerce").clip(lower=1.0)
    if max_col in df.columns:
        df["max_avg_spread"] = np.log(
            pd.to_numeric(df[max_col], errors="coerce").clip(lower=1.0) / raw_price
        )
    else:
        df["max_avg_spread"] = np.nan
    added.append("max_avg_spread")

    # 8. Today's auctioneer composition effect (expected to revert on next session)
    if "house_effect_log" not in df.columns:
        df["house_effect_log"] = 0.0
    else:
        df["house_effect_log"] = pd.to_numeric(df["house_effect_log"], errors="coerce").fillna(0.0)
    added.append("house_effect_log")

    return added


# ── Tier 1b: Auction microstructure ──────────────────────────────────────

MICRO_FEATURES = [
    "disp_cv_now", "disp_cv_ma", "range_pct_now",
    "unsold_now", "unsold_ma",
    "n_auctions_ma", "lot_size_rel", "grade_spread_ma",
]


def add_micro(df: pd.DataFrame, freq: str = "daily") -> list[str]:
    """Add causal microstructure features aligned to row `t`."""
    cfg = _FREQ_WINDOWS[freq]
    m_w = cfg["micro_ma"]
    l_w = cfg["lot_base"]
    added = []

    src = [
        "price_disp_cv", "price_range_pct", "unsold_pct",
        "n_auctions", "avg_lot_kg", "grade_spread_pct",
    ]
    present = [c for c in src if c in df.columns]
    if present:
        df[present] = df[present].ffill()

    if "price_disp_cv" in df.columns:
        df["disp_cv_now"] = df["price_disp_cv"]
        df["disp_cv_ma"] = df["price_disp_cv"].rolling(m_w, min_periods=max(2, m_w // 2)).mean()
        added.extend(["disp_cv_now", "disp_cv_ma"])

    if "price_range_pct" in df.columns:
        df["range_pct_now"] = df["price_range_pct"]
        added.append("range_pct_now")

    if "unsold_pct" in df.columns:
        df["unsold_now"] = df["unsold_pct"]
        df["unsold_ma"] = df["unsold_pct"].rolling(m_w, min_periods=max(2, m_w // 2)).mean()
        added.extend(["unsold_now", "unsold_ma"])

    if "n_auctions" in df.columns:
        df["n_auctions_ma"] = df["n_auctions"].rolling(m_w, min_periods=max(2, m_w // 2)).mean()
        added.append("n_auctions_ma")

    if "avg_lot_kg" in df.columns:
        base = df["avg_lot_kg"].rolling(l_w, min_periods=max(4, l_w // 2)).mean()
        df["lot_size_rel"] = np.log((df["avg_lot_kg"].clip(lower=1.0)) / base.clip(lower=1.0))
        added.append("lot_size_rel")

    if "grade_spread_pct" in df.columns:
        df["grade_spread_ma"] = df["grade_spread_pct"].rolling(
            m_w, min_periods=max(2, m_w // 2)
        ).mean()
        added.append("grade_spread_ma")

    return added


# ── Tier 2: Calendar / Seasonal ──────────────────────────────────────────

T2_FEATURES_DAILY = [
    "month_sin", "month_cos", "week_sin", "week_cos", "dow_sin", "dow_cos",
    "wedding_season", "harvest_season", "peak_harvest",
    "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    "is_lean", "is_strong",
]

T2_FEATURES_WEEKLY = [
    "month_sin", "month_cos", "week_sin", "week_cos",
    "wedding_season", "harvest_season", "peak_harvest",
    "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    "is_lean", "is_strong",
]

T2_FEATURES_MONTHLY = ["month_sin", "month_cos", "is_lean", "is_strong"]


def add_tier2(df: pd.DataFrame, freq: str = "daily", date_col: str = "date") -> list[str]:
    """Add T2 calendar/seasonal features."""
    dates = pd.to_datetime(df[date_col])
    added = []

    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12.0)
    added.extend(["month_sin", "month_cos"])

    if freq in ("daily", "weekly"):
        week = dates.dt.isocalendar().week.astype(int)
        df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * week / 52.0)
        added.extend(["week_sin", "week_cos"])

    if freq == "daily":
        dow = dates.dt.dayofweek.astype(float)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 6.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 6.0)
        added.extend(["dow_sin", "dow_cos"])

    month = dates.dt.month
    df["is_lean"] = ((month >= 3) & (month <= 5)).astype(int)
    df["is_strong"] = month.isin([8, 9, 12, 1]).astype(int)
    added.extend(["is_lean", "is_strong"])

    for col in [
        "wedding_season", "harvest_season", "peak_harvest",
        "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    ]:
        if col in df.columns:
            added.append(col)

    return added


# ── Tier 3: Frequency-Aware Weather / ENSO / Forward Forecast ────────────

T3_FEATURES = [
    "rain_cum_1", "rain_cum_2", "rain_cum_3", "rain_cum_4", "rain_anomaly",
    "gt_rain_cum_1", "gt_rain_cum_2", "gt_rain_cum_3",
    "temp_ma", "humidity_ma",
    "enso", "enso_lag6m", "enso_lag12m", "enso_phase",
]

T3_FORECAST_FEATURES = [
    "fcst_rain_14d_anom", "fcst_gt_rain_14d_anom",
]


def add_tier3(df: pd.DataFrame, freq: str = "daily") -> list[str]:
    """Add frequency-parameterized T3 weather/ENSO features (strictly causal)."""
    cfg = _FREQ_WINDOWS[freq]
    added = []
    # External daily weather arrives with a 1-day lag; weekly/monthly bars already close the period
    lag = 1 if freq == "daily" else 0

    if "rain_mm" in df.columns:
        rain = pd.to_numeric(df["rain_mm"], errors="coerce").shift(lag)
        for idx, w in enumerate(cfg["rain"], start=1):
            col = f"rain_cum_{idx}"
            df[col] = rain.rolling(w, min_periods=max(1, w // 2)).sum()
            added.append(col)

        aw = cfg["rain_anom_win"]
        ann = cfg["rain_ann_win"]
        recent_sum = rain.rolling(aw, min_periods=max(1, aw // 2)).sum()
        ann_mean_per_bar = rain.rolling(ann, min_periods=max(3, ann // 2)).mean()
        # Same unit comparison: recent `aw`-bar sum minus expected `aw`-bar sum
        df["rain_anomaly"] = (recent_sum - ann_mean_per_bar * aw) / (ann_mean_per_bar * aw + 25.0)
        added.append("rain_anomaly")

    if "gt_rain_mm" in df.columns:
        gt_rain = pd.to_numeric(df["gt_rain_mm"], errors="coerce").shift(lag)
        for idx, w in enumerate(cfg["gt_rain"], start=1):
            col = f"gt_rain_cum_{idx}"
            df[col] = gt_rain.rolling(w, min_periods=max(1, w // 2)).sum()
            added.append(col)

    if "temp_max_c" in df.columns:
        tw = cfg["temp_ma"]
        df["temp_ma"] = pd.to_numeric(df["temp_max_c"], errors="coerce").shift(lag).rolling(
            tw, min_periods=max(1, tw // 2)
        ).mean()
        added.append("temp_ma")

    if "humidity_pct" in df.columns:
        tw = cfg["temp_ma"]
        df["humidity_ma"] = pd.to_numeric(df["humidity_pct"], errors="coerce").shift(lag).rolling(
            tw, min_periods=max(1, tw // 2)
        ).mean()
        added.append("humidity_ma")

    if "ENSO" in df.columns:
        enso_s = pd.to_numeric(df["ENSO"], errors="coerce")
        df["enso"] = enso_s
        df["enso_lag6m"] = enso_s.shift(cfg["enso_6m"])
        df["enso_lag12m"] = enso_s.shift(cfg["enso_12m"])
        df["enso_phase"] = 0
        df.loc[df["enso"] > 0.5, "enso_phase"] = 1
        df.loc[df["enso"] < -0.5, "enso_phase"] = -1
        added.extend(["enso", "enso_lag6m", "enso_lag12m", "enso_phase"])

    # Forward-looking 14-day forecast anomalies (when live forecast rows exist)
    if "idukki_fcst_rain_14d" in df.columns and "rain_mm" in df.columns:
        base_14 = pd.to_numeric(df["rain_mm"], errors="coerce").rolling(
            cfg["rain_ann_win"], min_periods=max(3, cfg["rain_ann_win"] // 2)
        ).mean() * (14.0 if freq == "daily" else (2.0 if freq == "weekly" else 0.5))
        fcst = pd.to_numeric(df["idukki_fcst_rain_14d"], errors="coerce")
        df["fcst_rain_14d_anom"] = (fcst - base_14) / (base_14 + 25.0)
        added.append("fcst_rain_14d_anom")

    if "gt_fcst_rain_14d" in df.columns and "gt_rain_mm" in df.columns:
        base_14 = pd.to_numeric(df["gt_rain_mm"], errors="coerce").rolling(
            cfg["rain_ann_win"], min_periods=max(3, cfg["rain_ann_win"] // 2)
        ).mean() * (14.0 if freq == "daily" else (2.0 if freq == "weekly" else 0.5))
        fcst = pd.to_numeric(df["gt_fcst_rain_14d"], errors="coerce")
        df["fcst_gt_rain_14d_anom"] = (fcst - base_14) / (base_14 + 25.0)
        added.append("fcst_gt_rain_14d_anom")

    return added


# ── Tier 4: Stationary Macro / Financial (no raw FX levels) ──────────────

T4_FEATURES = [
    "usdinr_pct_short", "usdinr_pct_long",
    "crudeoil_pct_short", "gold_pct_short", "nifty_pct_short",
]


def add_tier4(df: pd.DataFrame, freq: str = "daily") -> list[str]:
    """Add stationary frequency-scaled T4 macro financial return features."""
    cfg = _FREQ_WINDOWS[freq]
    w_s, w_l = cfg["macro_short"], cfg["macro_long"]
    lag = 1 if freq == "daily" else 0
    added = []

    if "usdinr" in df.columns:
        fx = pd.to_numeric(df["usdinr"], errors="coerce").shift(lag)
        df["usdinr_pct_short"] = fx.pct_change(w_s) * 100.0
        df["usdinr_pct_long"] = fx.pct_change(w_l) * 100.0
        added.extend(["usdinr_pct_short", "usdinr_pct_long"])

    for name, col in [("crudeoil", "crude_oil"), ("gold", "gold"), ("nifty", "nifty")]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").shift(lag)
            feat = f"{name}_pct_short"
            df[feat] = s.pct_change(w_s) * 100.0
            added.append(feat)

    return added


# ── Tier 5: Trade / Supply (precomputed on native monthly trade tables) ──

T5_FEATURES = ["gt_qty_3m_log", "gt_qty_yoy", "gt_unit_price_log", "sa_qty_3m_log", "sa_qty_yoy"]


def add_tier5(df: pd.DataFrame, freq: str = "monthly") -> list[str]:
    """Expose/compute native-monthly T5 trade features without frequency distortion."""
    added = []

    # assemble.py precomputes gt_qty_3m, gt_qty_yoy, gt_unit_price on the native
    # monthly Comtrade table BEFORE backward merging. Fallback only if called standalone.
    if "gt_qty_3m" not in df.columns and "gt_qty_kg" in df.columns and freq == "monthly":
        qty_mt = pd.to_numeric(df["gt_qty_kg"], errors="coerce") / 1000.0
        df["gt_qty_3m"] = qty_mt.rolling(3, min_periods=1).sum()
        df["gt_qty_yoy"] = (qty_mt / qty_mt.shift(12).clip(lower=1e-3) - 1.0).clip(-0.95, 3.0) * 100.0
        if "gt_value_usd" in df.columns:
            df["gt_unit_price"] = df["gt_value_usd"] / df["gt_qty_kg"].clip(lower=1.0)

    if "gt_qty_3m" in df.columns:
        df["gt_qty_3m_log"] = np.log1p(pd.to_numeric(df["gt_qty_3m"], errors="coerce").clip(lower=0.0))
        added.append("gt_qty_3m_log")
    if "gt_qty_yoy" in df.columns:
        added.append("gt_qty_yoy")
    if "gt_unit_price" in df.columns:
        df["gt_unit_price_log"] = np.log(pd.to_numeric(df["gt_unit_price"], errors="coerce").clip(lower=0.5))
        added.append("gt_unit_price_log")

    if "sa_qty_3m" not in df.columns and "sa_qty_kg" in df.columns and freq == "monthly":
        qty_mt = pd.to_numeric(df["sa_qty_kg"], errors="coerce") / 1000.0
        df["sa_qty_3m"] = qty_mt.rolling(3, min_periods=1).sum()
        df["sa_qty_yoy"] = (qty_mt / qty_mt.shift(12).clip(lower=1e-3) - 1.0).clip(-0.95, 3.0) * 100.0

    if "sa_qty_3m" in df.columns:
        df["sa_qty_3m_log"] = np.log1p(pd.to_numeric(df["sa_qty_3m"], errors="coerce").clip(lower=0.0))
        added.append("sa_qty_3m_log")
    if "sa_qty_yoy" in df.columns:
        added.append("sa_qty_yoy")

    return added


# ── Tier 6: Structural / Cycle ───────────────────────────────────────────

T6_FEATURES = [
    "log_price_to_cost", "months_since_trough", "cycle_age_norm",
    "gt_plantation_ma12", "gt_price_interest", "vol_vs_2yr",
]


def add_tier6(df: pd.DataFrame, freq: str = "monthly", price_col: str | None = None) -> list[str]:
    """Add T6 structural/cycle features (causal at t)."""
    cfg = _FREQ_WINDOWS[freq]
    if price_col is None:
        price_col = "adj_price" if "adj_price" in df.columns else "avg_price"

    added = []
    dates = pd.to_datetime(df["date"])

    # Cost floor (MGNREGA-based) — log ratio is symmetric and regime-stable
    base_wage = 212.0
    cagr = 0.052
    years_from_base = (dates - pd.Timestamp("2014-01-01")).dt.days / 365.25
    mgnrega_wage = base_wage * (1.0 + cagr) ** years_from_base
    cost_per_kg = mgnrega_wage * 1.7 * 200.0 / 0.62 / 350.0
    df["cost_floor"] = cost_per_kg
    price = pd.to_numeric(df[price_col], errors="coerce").clip(lower=1.0)
    df["price_to_cost"] = price / cost_per_kg
    df["log_price_to_cost"] = np.log(df["price_to_cost"].clip(lower=0.1))
    added.append("log_price_to_cost")

    # Causal cobweb cycle: months since confirmed trailing price trough
    win = min(18, max(6, len(df) // 6)) if freq == "monthly" else min(180, max(30, len(df) // 4))
    smoothed = price.rolling(win, min_periods=max(3, win // 2)).mean()

    if len(df) > 24:
        w = min(6, max(2, len(df) // 12))
        trough_idx = []
        for i in range(w, len(df) - w):
            window = smoothed.iloc[i - w: i + w + 1]
            if not pd.isna(smoothed.iloc[i]) and smoothed.iloc[i] == window.min():
                trough_idx.append(i)

        if trough_idx:
            def _months_since(j):
                past = [i for i in trough_idx if i + w <= j]
                if not past:
                    return np.nan
                return (dates.iloc[j] - dates.iloc[past[-1]]).days / 30.44

            df["months_since_trough"] = [_months_since(j) for j in range(len(df))]
            df["cycle_age_norm"] = df["months_since_trough"] / 48.0
            added.extend(["months_since_trough", "cycle_age_norm"])

    # Google Trends
    if "cardamom_plantation" in df.columns:
        gt_w = 12 if freq == "monthly" else (52 if freq == "weekly" else 250)
        df["gt_plantation_ma12"] = df["cardamom_plantation"].rolling(
            gt_w, min_periods=max(3, gt_w // 2)
        ).mean()
        added.append("gt_plantation_ma12")
    if "cardamom_price" in df.columns:
        df["gt_price_interest"] = df["cardamom_price"]
        added.append("gt_price_interest")

    # Volume vs 2-year average (frequency-scaled)
    if "sold_kg" in df.columns and len(df) > 24:
        vol = pd.to_numeric(df["sold_kg"], errors="coerce").clip(lower=0.0)
        vs, vl = cfg["vol_short"], cfg["vol_2yr"]
        df["vol_vs_2yr"] = np.log(
            (vol.rolling(vs, min_periods=max(1, vs // 2)).mean() + 1.0)
            / (vol.rolling(vl, min_periods=max(6, vl // 2)).mean() + 1.0)
        )
        added.append("vol_vs_2yr")

    return added


# ── Master builders ──────────────────────────────────────────────────────

def build_daily_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply all applicable tiers to a daily DataFrame."""
    feats = []
    feats += add_tier1(df, freq="daily")
    feats += add_micro(df, freq="daily")
    feats += add_tier2(df, freq="daily")
    feats += add_tier3(df, freq="daily")
    feats += add_tier4(df, freq="daily")
    return df, feats


def build_weekly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply frequency-aware tiers to weekly DataFrame."""
    feats = []
    feats += add_tier1(df, freq="weekly", vol_col="sold_kg", arr_col="arrived_kg")
    feats += add_micro(df, freq="weekly")
    feats += add_tier2(df, freq="weekly")
    feats += add_tier3(df, freq="weekly")
    feats += add_tier4(df, freq="weekly")
    feats += add_tier5(df, freq="weekly")
    return df, feats


def build_monthly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply frequency-aware tiers to monthly DataFrame for 90-day + regime models."""
    feats = []
    feats += add_tier1(df, freq="monthly", vol_col="sold_kg", arr_col="arrived_kg")
    feats += add_tier2(df, freq="monthly")
    feats += add_tier3(df, freq="monthly")
    feats += add_tier4(df, freq="monthly")
    feats += add_tier5(df, freq="monthly")
    feats += add_tier6(df, freq="monthly")
    return df, feats
