"""
features.py — Zone-to-Zone Feature Engineering

Key fix:
  detect_zones() now tracks a LIST of active demand and supply zones,
  not a single variable. This means when price is in a demand zone,
  the nearest supply zone above is always available for TP calculation.
  Zones are invalidated when price closes through them (broken zone).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.features")

DAY_OF_WEEK_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Multi-Zone Tracker
# ---------------------------------------------------------------------------

def detect_zones(
    df: pd.DataFrame,
    lookback: int = 30,
    impulse_atr_multiplier: float = 0.5,
    atr_period: int = 14,
    max_zones: int = 5,
) -> pd.DataFrame:
    """
    Detect supply and demand zones using a zone LIST (not single variable).

    For each bar:
      - Checks if any existing zones are broken (price closed through) → remove
      - Detects new zones from impulse candles
      - Finds the nearest demand zone BELOW price → demand context
      - Finds the nearest supply zone ABOVE price → supply context
      - Both are written to the row simultaneously

    This means TP is always calculable when price is in a zone.
    """
    df   = df.copy().reset_index(drop=True)
    atr  = _atr(df, atr_period)

    zone_cols = [
        "demand_zone_top", "demand_zone_bottom", "demand_zone_strength",
        "demand_zone_fresh", "demand_zone_touches",
        "supply_zone_top", "supply_zone_bottom", "supply_zone_strength",
        "supply_zone_fresh", "supply_zone_touches",
        "nearest_demand_dist_atr", "nearest_supply_dist_atr",
        "in_demand_zone", "in_supply_zone", "between_zones",
    ]
    for col in zone_cols:
        df[col] = np.nan

    # Zone list entries: dict with top, bottom, strength, touches, fresh, type
    demand_zones = []  # list of active demand zones
    supply_zones = []  # list of active supply zones

    for i in range(lookback, len(df)):
        cur     = df.iloc[i]
        close   = float(cur["close"])
        high    = float(cur["high"])
        low     = float(cur["low"])
        cur_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 1.0
        body    = float(abs(cur["close"] - cur["open"]))

        # ----------------------------------------------------------------
        # Invalidate broken zones — price closed through them
        # ----------------------------------------------------------------
        demand_zones = [z for z in demand_zones if close > z["bottom"]]  # keep if price above bottom
        supply_zones = [z for z in supply_zones if close < z["top"]]     # keep if price below top

        # ----------------------------------------------------------------
        # Detect new demand zone (bullish impulse)
        # ----------------------------------------------------------------
        if cur["close"] > cur["open"] and body > impulse_atr_multiplier * cur_atr:
            new_zone = {
                "top":      float(max(cur["open"], cur["close"])),
                "bottom":   float(cur["low"]),
                "strength": float(min(body / cur_atr, 5.0)),
                "touches":  0,
                "fresh":    True,
            }
            demand_zones.append(new_zone)
            # Keep only the strongest max_zones zones
            if len(demand_zones) > max_zones:
                demand_zones = sorted(demand_zones, key=lambda z: z["strength"], reverse=True)[:max_zones]

        # ----------------------------------------------------------------
        # Detect new supply zone (bearish impulse)
        # ----------------------------------------------------------------
        if cur["close"] < cur["open"] and body > impulse_atr_multiplier * cur_atr:
            new_zone = {
                "top":      float(cur["high"]),
                "bottom":   float(min(cur["open"], cur["close"])),
                "strength": float(min(body / cur_atr, 5.0)),
                "touches":  0,
                "fresh":    True,
            }
            supply_zones.append(new_zone)
            if len(supply_zones) > max_zones:
                supply_zones = sorted(supply_zones, key=lambda z: z["strength"], reverse=True)[:max_zones]

        # ----------------------------------------------------------------
        # Find nearest demand zone (highest bottom below close)
        # ----------------------------------------------------------------
        demand_below = [z for z in demand_zones if z["top"] <= close]
        nearest_demand = None
        if demand_below:
            nearest_demand = max(demand_below, key=lambda z: z["top"])  # closest below

        # ----------------------------------------------------------------
        # Find nearest supply zone (lowest top above close)
        # ----------------------------------------------------------------
        supply_above = [z for z in supply_zones if z["bottom"] >= close]
        nearest_supply = None
        if supply_above:
            nearest_supply = min(supply_above, key=lambda z: z["bottom"])  # closest above

        # ----------------------------------------------------------------
        # Write demand zone to row
        # ----------------------------------------------------------------
        in_demand = False
        if nearest_demand is not None:
            df.at[i, "demand_zone_top"]      = nearest_demand["top"]
            df.at[i, "demand_zone_bottom"]   = nearest_demand["bottom"]
            df.at[i, "demand_zone_strength"] = nearest_demand["strength"]
            df.at[i, "demand_zone_fresh"]    = float(nearest_demand["fresh"])
            df.at[i, "demand_zone_touches"]  = float(nearest_demand["touches"])
            dist = (close - nearest_demand["top"]) / cur_atr
            df.at[i, "nearest_demand_dist_atr"] = float(dist)

            # In demand zone if price low is within the zone
            in_demand = nearest_demand["bottom"] <= low <= nearest_demand["top"]
            df.at[i, "in_demand_zone"] = float(in_demand)
            if in_demand:
                nearest_demand["touches"] += 1
                nearest_demand["fresh"]    = False
        else:
            df.at[i, "in_demand_zone"] = 0.0

        # ----------------------------------------------------------------
        # Write supply zone to row
        # ----------------------------------------------------------------
        in_supply = False
        if nearest_supply is not None:
            df.at[i, "supply_zone_top"]      = nearest_supply["top"]
            df.at[i, "supply_zone_bottom"]   = nearest_supply["bottom"]
            df.at[i, "supply_zone_strength"] = nearest_supply["strength"]
            df.at[i, "supply_zone_fresh"]    = float(nearest_supply["fresh"])
            df.at[i, "supply_zone_touches"]  = float(nearest_supply["touches"])
            dist = (nearest_supply["bottom"] - close) / cur_atr
            df.at[i, "nearest_supply_dist_atr"] = float(dist)

            # In supply zone if price high is within the zone
            in_supply = nearest_supply["bottom"] <= high <= nearest_supply["top"]
            df.at[i, "in_supply_zone"] = float(in_supply)
            if in_supply:
                nearest_supply["touches"] += 1
                nearest_supply["fresh"]    = False
        else:
            df.at[i, "in_supply_zone"] = 0.0

        # Between zones — price is between demand top and supply bottom
        if nearest_demand is not None and nearest_supply is not None:
            between = (
                nearest_demand["top"] < close < nearest_supply["bottom"]
            )
            df.at[i, "between_zones"] = float(between)
        else:
            df.at[i, "between_zones"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Confirmation Signals
# ---------------------------------------------------------------------------

def add_confirmation_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    body       = (df["close"] - df["open"]).abs()
    wick_upper = df["high"]  - df[["open", "close"]].max(axis=1)
    wick_lower = df[["open", "close"]].min(axis=1) - df["low"]

    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    df["bullish_engulfing"] = (
        (prev_close < prev_open) &
        (df["close"] > df["open"]) &
        (df["open"]  < prev_close) &
        (df["close"] > prev_open)
    ).astype(float)

    df["bearish_engulfing"] = (
        (prev_close > prev_open) &
        (df["close"] < df["open"]) &
        (df["open"]  > prev_close) &
        (df["close"] < prev_open)
    ).astype(float)

    safe_body = body.clip(lower=1e-10)
    df["pin_bar_bullish"] = (
        (wick_lower > 2.0 * safe_body) &
        (wick_lower > 2.0 * wick_upper)
    ).astype(float)

    df["pin_bar_bearish"] = (
        (wick_upper > 2.0 * safe_body) &
        (wick_upper > 2.0 * wick_lower)
    ).astype(float)

    df["higher_low"] = (df["low"]  > df["low"].shift(1)).astype(float)
    df["lower_high"] = (df["high"] < df["high"].shift(1)).astype(float)

    swing_high = df["high"].rolling(5).max().shift(1)
    swing_low  = df["low"].rolling(5).min().shift(1)
    df["bos_bullish"] = (df["close"] > swing_high).astype(float)
    df["bos_bearish"] = (df["close"] < swing_low).astype(float)

    df["buy_confirmation_score"]  = (
        df["bullish_engulfing"] + df["pin_bar_bullish"] +
        df["higher_low"]        + df["bos_bullish"]
    )
    df["sell_confirmation_score"] = (
        df["bearish_engulfing"] + df["pin_bar_bearish"] +
        df["lower_high"]        + df["bos_bearish"]
    )

    return df


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["atr_14"]  = _atr(df, 14)
    df["rsi_14"]  = _rsi(df["close"], 14)
    df["ema_20"]  = _ema(df["close"], 20)
    df["ema_50"]  = _ema(df["close"], 50)
    df["ema_200"] = _ema(df["close"], 200)

    safe_atr = df["atr_14"].replace(0, np.nan)

    if "spread" in df.columns:
        df["spread_norm"] = df["spread"] / safe_atr
    else:
        df["spread_norm"] = 0.0

    df["ema_spread_atr"]     = (df["ema_20"] - df["ema_50"]) / safe_atr
    df["price_above_ema20"]  = (df["close"] > df["ema_20"]).astype(float)
    df["price_above_ema50"]  = (df["close"] > df["ema_50"]).astype(float)
    df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(float)

    df["ema_trend_bias"] = np.where(
        (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]),  1,
        np.where(
            (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]), -1, 0
        )
    ).astype(float)

    bb_mid   = df["close"].rolling(20).mean()
    bb_std   = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"]  = (df["close"] - bb_lower) / bb_width
    df["bb_width_atr"] = bb_width / safe_atr

    vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = df["volume"] / vol_ma
    df["momentum_5"]   = (df["close"] - df["close"].shift(5))  / safe_atr
    df["momentum_10"]  = (df["close"] - df["close"].shift(10)) / safe_atr

    # ATR-normalized candle metrics
    df["candle_size_atr"] = df["candle_size"] / safe_atr
    df["body_size_atr"]   = df["body_size"]   / safe_atr
    df["wick_upper_atr"]  = df["wick_upper"]  / safe_atr
    df["wick_lower_atr"]  = df["wick_lower"]  / safe_atr
    df["body_atr_ratio"]  = df["body_size"]   / safe_atr

    if "day_of_week" in df.columns:
        df["day_of_week_int"] = df["day_of_week"].map(DAY_OF_WEEK_MAP).fillna(-1).astype(int)
    else:
        df["day_of_week_int"] = -1

    return df


# ---------------------------------------------------------------------------
# HTF Context
# ---------------------------------------------------------------------------

def add_htf_context(
    ltf_df: pd.DataFrame,
    h1_df:  pd.DataFrame,
    h4_df:  pd.DataFrame,
) -> pd.DataFrame:
    ltf_df = ltf_df.copy()
    ltf_df["timestamp"] = pd.to_datetime(ltf_df["timestamp"])

    def _bias(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        atr  = _atr(df, 14)
        body = (df["close"] - df["open"]).abs()
        impulse = body > 0.5 * atr
        bias = np.where(impulse & (df["close"] > df["open"]),  1,
               np.where(impulse & (df["close"] < df["open"]), -1, 0))
        df["htf_bias"] = pd.Series(bias, index=df.index)
        df["htf_bias"] = df["htf_bias"].replace(0, np.nan).ffill().fillna(0).astype(float)
        return df[["timestamp", "htf_bias"]]

    if h1_df is not None and len(h1_df) > 0:
        h1b = _bias(h1_df).rename(columns={"htf_bias": "htf_1h_bias"})
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h1b.sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
    else:
        ltf_df["htf_1h_bias"] = 0.0

    if h4_df is not None and len(h4_df) > 0:
        h4b = _bias(h4_df).rename(columns={"htf_bias": "htf_4h_bias"})
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h4b.sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
    else:
        ltf_df["htf_4h_bias"] = 0.0

    ltf_df["htf_aligned"] = (
        (ltf_df["htf_1h_bias"] == ltf_df["htf_4h_bias"]) &
        (ltf_df["htf_1h_bias"] != 0)
    ).astype(float)

    return ltf_df


# ---------------------------------------------------------------------------
# Master Builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
    zone_lookback: int = 30,
    impulse_atr_multiplier: float = 0.5,
) -> pd.DataFrame:
    logger.info(f"Building features for {len(df)} rows...")

    df = detect_zones(df, lookback=zone_lookback, impulse_atr_multiplier=impulse_atr_multiplier)
    df = add_confirmation_signals(df)
    df = add_indicators(df)

    if h1_df is not None or h4_df is not None:
        df = add_htf_context(df, h1_df, h4_df)

    warmup = max(200, zone_lookback)
    df = df.iloc[warmup:].reset_index(drop=True)

    logger.info(f"Features built — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature columns for ML
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Zone
    "demand_zone_strength", "demand_zone_fresh", "demand_zone_touches",
    "supply_zone_strength", "supply_zone_fresh", "supply_zone_touches",
    "nearest_demand_dist_atr", "nearest_supply_dist_atr",
    "in_demand_zone", "in_supply_zone", "between_zones",
    # Confirmations
    "bullish_engulfing", "bearish_engulfing",
    "pin_bar_bullish", "pin_bar_bearish",
    "higher_low", "lower_high",
    "bos_bullish", "bos_bearish",
    "buy_confirmation_score", "sell_confirmation_score",
    # Indicators
    "atr_14", "rsi_14",
    "ema_spread_atr",
    "price_above_ema20", "price_above_ema50", "price_above_ema200",
    "ema_trend_bias",
    "bb_position", "bb_width_atr",
    "volume_ratio", "body_atr_ratio",
    "momentum_5", "momentum_10",
    # HTF
    "htf_1h_bias", "htf_4h_bias", "htf_aligned",
    # Candle context — ATR normalized
    "candle_size_atr", "body_size_atr", "wick_upper_atr", "wick_lower_atr",
    # Temporal
    "hour", "month", "day_of_week_int",
    # Spread
    "spread_norm",
]