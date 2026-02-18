"""Feature engineering: 6 tiers matching the notebook framework."""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    return 100 - (100 / (1 + rs))


# ── Tier 1: Technical / Momentum (33 features) ──────────────────────────

T1_FEATURES = (
    [f"lag_{l}" for l in [1, 2, 3, 5, 7, 14, 21, 30]]
    + [f"ma_{w}" for w in [7, 14, 30, 60, 90]]
    + [f"std_{w}" for w in [7, 14, 30]]
    + [f"min_{w}" for w in [14, 30]]
    + [f"max_{w}" for w in [14, 30]]
    + [f"pct_{w}" for w in [7, 14, 30]]
    + ["rsi_14", "boll_pos", "price_vs_ma30", "price_vs_ma90"]
    + [f"vol_ma_{w}" for w in [7, 14, 30]]
    + ["vol_ratio", "sell_through_14", "max_avg_spread"]
)


def add_tier1(df: pd.DataFrame, price_col: str = "avg_price",
              vol_col: str = "sold_kg", arr_col: str = "arrived_kg",
              max_col: str = "max_price") -> list[str]:
    """Add T1 technical/momentum features. Returns list of feature names added."""
    price = df[price_col]
    added = []

    # Price lags
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        col = f"lag_{lag}"
        df[col] = price.shift(lag)
        added.append(col)

    # Moving averages
    for w in [7, 14, 30, 60, 90]:
        col = f"ma_{w}"
        df[col] = price.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        added.append(col)

    # Standard deviations
    for w in [7, 14, 30]:
        col = f"std_{w}"
        df[col] = price.shift(1).rolling(w, min_periods=max(1, w // 2)).std()
        added.append(col)

    # Min / Max
    for w in [14, 30]:
        df[f"min_{w}"] = price.shift(1).rolling(w).min()
        df[f"max_{w}"] = price.shift(1).rolling(w).max()
        added.extend([f"min_{w}", f"max_{w}"])

    # Percent changes
    for w in [7, 14, 30]:
        col = f"pct_{w}"
        df[col] = price.shift(1).pct_change(w) * 100
        added.append(col)

    # RSI
    df["rsi_14"] = compute_rsi(price.shift(1), 14)
    added.append("rsi_14")

    # Bollinger band position
    ma_30 = df.get("ma_30", price.shift(1).rolling(30, min_periods=15).mean())
    std_30 = df.get("std_30", price.shift(1).rolling(30, min_periods=15).std())
    df["boll_pos"] = (price.shift(1) - ma_30) / (2 * std_30.clip(lower=0.01))
    added.append("boll_pos")

    # Mean reversion
    ma_90 = df.get("ma_90", price.shift(1).rolling(90, min_periods=45).mean())
    df["price_vs_ma30"] = price.shift(1) / ma_30.clip(lower=1) - 1
    df["price_vs_ma90"] = price.shift(1) / ma_90.clip(lower=1) - 1
    added.extend(["price_vs_ma30", "price_vs_ma90"])

    # Volume features
    if vol_col in df.columns:
        volume = df[vol_col]
        for w in [7, 14, 30]:
            col = f"vol_ma_{w}"
            df[col] = volume.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
            added.append(col)
        df["vol_ratio"] = volume.shift(1) / df["vol_ma_30"].clip(lower=1)
        added.append("vol_ratio")
    else:
        for w in [7, 14, 30]:
            df[f"vol_ma_{w}"] = np.nan
        df["vol_ratio"] = np.nan
        added.extend([f"vol_ma_{w}" for w in [7, 14, 30]] + ["vol_ratio"])

    # Sell-through ratio
    if arr_col in df.columns and vol_col in df.columns:
        arrived = df[arr_col]
        volume = df[vol_col]
        df["sell_through_14"] = (
            volume.shift(1).rolling(14).sum()
            / arrived.shift(1).rolling(14).sum().clip(lower=1)
        )
    else:
        df["sell_through_14"] = np.nan
    added.append("sell_through_14")

    # Max-Avg spread
    if max_col in df.columns:
        df["max_avg_spread"] = df[max_col].shift(1) / price.shift(1) - 1
    else:
        df["max_avg_spread"] = np.nan
    added.append("max_avg_spread")

    return added


# ── Tier 2: Calendar / Seasonal ──────────────────────────────────────────

T2_FEATURES_DAILY = [
    "month_sin", "month_cos", "week_sin", "week_cos",
    "wedding_season", "harvest_season", "peak_harvest",
    "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear",
    "is_lean", "is_strong",
]

T2_FEATURES_MONTHLY = ["month_sin", "month_cos", "is_lean", "is_strong"]


def add_tier2(df: pd.DataFrame, date_col: str = "date") -> list[str]:
    """Add T2 calendar/seasonal features."""
    dates = pd.to_datetime(df[date_col])
    added = []

    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    added.extend(["month_sin", "month_cos"])

    if len(df) > 100:  # daily or weekly granularity
        week = dates.dt.isocalendar().week.astype(int)
        df["week_sin"] = np.sin(2 * np.pi * week / 52)
        df["week_cos"] = np.cos(2 * np.pi * week / 52)
        added.extend(["week_sin", "week_cos"])

    month = dates.dt.month
    df["is_lean"] = ((month >= 3) & (month <= 5)).astype(int)
    df["is_strong"] = month.isin([8, 9, 12, 1]).astype(int)
    added.extend(["is_lean", "is_strong"])

    # Festival flags (already in DataFrame from merge)
    for col in ["wedding_season", "harvest_season", "peak_harvest",
                "pre_eid_period", "pre_diwali_period", "pre_onam_period", "xmas_newyear"]:
        if col in df.columns:
            added.append(col)

    return added


# ── Tier 3: Weather / ENSO (14 features) ────────────────────────────────

T3_FEATURES = [
    "rain_cum_28", "rain_cum_56", "rain_cum_91", "rain_cum_182", "rain_anomaly",
    "gt_rain_cum_28", "gt_rain_cum_91", "gt_rain_cum_182",
    "temp_ma_28", "humidity_ma_28",
    "enso", "enso_lag6m", "enso_lag12m", "enso_phase",
]


def add_tier3(df: pd.DataFrame, is_monthly: bool = False) -> list[str]:
    """Add T3 weather/ENSO features."""
    added = []

    # Idukki rainfall
    if "rain_mm" in df.columns:
        for w in [28, 56, 91, 182]:
            col = f"rain_cum_{w}"
            min_p = max(1, w // 4) if is_monthly else max(1, w // 2)
            df[col] = df["rain_mm"].shift(1).rolling(w, min_periods=min_p).sum()
            added.append(col)
        rain_28 = df["rain_mm"].shift(1).rolling(28, min_periods=3 if is_monthly else 14).sum()
        annual_win = 12 if is_monthly else 365
        annual_min = 6 if is_monthly else 180
        rain_annual = df["rain_mm"].shift(1).rolling(annual_win, min_periods=annual_min).mean()
        ratio = (1 / 12) if is_monthly else (28 / 365)
        df["rain_anomaly"] = rain_28 - rain_annual * ratio
        added.append("rain_anomaly")

    # Guatemala rainfall
    if "gt_rain_mm" in df.columns:
        for w in [28, 91, 182]:
            col = f"gt_rain_cum_{w}"
            min_p = max(1, w // 4) if is_monthly else max(1, w // 2)
            df[col] = df["gt_rain_mm"].shift(1).rolling(w, min_periods=min_p).sum()
            added.append(col)

    # Temperature
    if "temp_max_c" in df.columns:
        df["temp_ma_28"] = df["temp_max_c"].shift(1).rolling(28, min_periods=14).mean()
        added.append("temp_ma_28")

    # Humidity
    if "humidity_pct" in df.columns:
        df["humidity_ma_28"] = df["humidity_pct"].shift(1).rolling(28, min_periods=14).mean()
        added.append("humidity_ma_28")

    # ENSO
    if "ENSO" in df.columns:
        df["enso"] = df["ENSO"].shift(1)
        added.append("enso")
        if is_monthly:
            df["enso_lag6m"] = df["ENSO"].shift(6)
            df["enso_lag12m"] = df["ENSO"].shift(12)
        else:
            df["enso_lag6m"] = df["ENSO"].shift(182)
            df["enso_lag12m"] = df["ENSO"].shift(365)
        added.extend(["enso_lag6m", "enso_lag12m"])

        df["enso_phase"] = 0
        df.loc[df["enso"] > 0.5, "enso_phase"] = 1     # El Nino
        df.loc[df["enso"] < -0.5, "enso_phase"] = -1    # La Nina
        added.append("enso_phase")

    return added


# ── Tier 4: Macro / Financial (6 features) ──────────────────────────────

T4_FEATURES = [
    "usdinr_lvl", "usdinr_pct_28", "usdinr_pct_90",
    "crudeoil_pct_28", "gold_pct_28", "nifty_pct_28",
]


def add_tier4(df: pd.DataFrame) -> list[str]:
    """Add T4 macro financial features."""
    added = []

    if "usdinr" in df.columns:
        df["usdinr_lvl"] = df["usdinr"].shift(1)
        df["usdinr_pct_28"] = df["usdinr"].shift(1).pct_change(28) * 100
        df["usdinr_pct_90"] = df["usdinr"].shift(1).pct_change(90) * 100
        added.extend(["usdinr_lvl", "usdinr_pct_28", "usdinr_pct_90"])

    for name, col in [("crudeoil", "crude_oil"), ("gold", "gold"), ("nifty", "nifty")]:
        if col in df.columns:
            feat = f"{name}_pct_28"
            df[feat] = df[col].shift(1).pct_change(28) * 100
            added.append(feat)

    return added


# ── Tier 5: Trade / Supply (5 features) ─────────────────────────────────

T5_FEATURES = ["gt_qty_3m", "gt_qty_yoy", "gt_unit_price", "sa_qty_3m", "sa_qty_yoy"]


def add_tier5(df: pd.DataFrame) -> list[str]:
    """Add T5 trade/supply features."""
    added = []

    if "gt_qty_kg" in df.columns:
        df["gt_qty_mt"] = df["gt_qty_kg"] / 1000
        df["gt_qty_3m"] = df["gt_qty_mt"].rolling(3, min_periods=1).sum()
        df["gt_qty_yoy"] = df["gt_qty_mt"].pct_change(12) * 100
        df["gt_unit_price"] = df["gt_value_usd"] / df["gt_qty_kg"].clip(lower=1)
        added.extend(["gt_qty_3m", "gt_qty_yoy", "gt_unit_price"])

    if "sa_qty_kg" in df.columns:
        df["sa_qty_mt"] = df["sa_qty_kg"] / 1000
        df["sa_qty_3m"] = df["sa_qty_mt"].rolling(3, min_periods=1).sum()
        df["sa_qty_yoy"] = df["sa_qty_mt"].pct_change(12) * 100
        added.extend(["sa_qty_3m", "sa_qty_yoy"])

    return added


# ── Tier 6: Structural / Cycle (6 features) ─────────────────────────────

T6_FEATURES = [
    "price_to_cost", "months_since_trough", "cycle_age_norm",
    "gt_plantation_ma12", "gt_price_interest", "vol_vs_2yr",
]


def add_tier6(df: pd.DataFrame, price_col: str = "avg_price") -> list[str]:
    """Add T6 structural/cycle features."""
    added = []
    dates = pd.to_datetime(df["date"])

    # Cost floor (MGNREGA-based)
    base_wage = 212
    cagr = 0.052
    years_from_base = (dates - pd.Timestamp("2014-01-01")).dt.days / 365.25
    mgnrega_wage = base_wage * (1 + cagr) ** years_from_base
    cost_per_kg = mgnrega_wage * 1.7 * 200 / 0.62 / 350
    df["cost_floor"] = cost_per_kg
    df["price_to_cost"] = df[price_col].shift(1) / cost_per_kg
    added.append("price_to_cost")

    # Cobweb cycle: months since price trough
    smoothed = df[price_col].rolling(
        min(180, max(30, len(df) // 4)), min_periods=30, center=True
    ).mean()

    if len(df) > 24:
        w = min(6, max(1, len(df) // 6))
        troughs = []
        for i in range(w, len(df) - w):
            window = smoothed.iloc[max(0, i - w): min(len(df), i + w + 1)]
            if not pd.isna(smoothed.iloc[i]) and smoothed.iloc[i] == window.min():
                troughs.append(dates.iloc[i])

        if troughs:
            def _months_since(x):
                past = [t for t in troughs if t <= x]
                return (x - past[-1]).days / 30.44 if past else np.nan

            df["months_since_trough"] = dates.apply(_months_since)
            df["cycle_age_norm"] = df["months_since_trough"] / 48
            added.extend(["months_since_trough", "cycle_age_norm"])

    # Google Trends
    if "cardamom_plantation" in df.columns:
        df["gt_plantation_ma12"] = df["cardamom_plantation"].rolling(12, min_periods=6).mean()
        added.append("gt_plantation_ma12")
    if "cardamom_price" in df.columns:
        df["gt_price_interest"] = df["cardamom_price"]
        added.append("gt_price_interest")

    # Volume vs 2-year average
    if "sold_kg" in df.columns and len(df) > 24:
        df["vol_vs_2yr"] = (
            df["sold_kg"].shift(1).rolling(3, min_periods=1).mean()
            / df["sold_kg"].shift(1).rolling(24, min_periods=12).mean().clip(lower=1)
        )
        added.append("vol_vs_2yr")

    return added


# ── Master builder ───────────────────────────────────────────────────────

def build_daily_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply all applicable tiers to a daily DataFrame. Returns (df, feature_names)."""
    feats = []
    feats += add_tier1(df)
    feats += add_tier2(df)
    feats += add_tier3(df, is_monthly=False)
    feats += add_tier4(df)
    return df, feats


def build_weekly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply tiers to weekly DataFrame."""
    feats = []
    feats += add_tier1(df, vol_col="sold_kg", arr_col="arrived_kg")
    feats += add_tier2(df)
    feats += add_tier3(df, is_monthly=False)
    feats += add_tier4(df)
    feats += add_tier5(df)
    return df, feats


def build_monthly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply tiers to monthly DataFrame for 90-day + regime models."""
    feats = []
    feats += add_tier1(df, vol_col="sold_kg", arr_col="arrived_kg")
    feats += add_tier2(df)
    feats += add_tier3(df, is_monthly=True)
    feats += add_tier5(df)
    feats += add_tier6(df)
    return df, feats
