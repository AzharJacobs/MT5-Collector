"""
Feature Engineering Module — Zone-to-Zone Strategy
====================================================
Builds ML-ready features from raw OHLCV data aligned to the Zone-to-Zone strategy:

Features computed:
  - Supply / Demand zone detection (strength score, freshness, distance)
  - HTF context: 1H and 4H zone bias (bullish / bearish / neutral)
  - Confirmation signals: bullish/bearish engulfing, pin bars, BOS, higher-low / lower-high
  - Technical indicators: ATR, RSI, EMA fast/slow, Bollinger Bands, volume momentum
  - Candle context: position relative to zone, session label

All features are computed on a rolling basis — NO lookahead.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("mt5_collector.features")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Zone Detection
# ---------------------------------------------------------------------------

def detect_zones(
    df: pd.DataFrame,
    lookback: int = 50,
    impulse_atr_multiplier: float = 1.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Detect supply and demand zones using explosive move logic from the strategy.

    A Demand Zone is identified when:
      - A strong bullish impulse candle appears (body > 1.5x ATR)
      - Preceded by consolidation (base candles with small bodies)

    A Supply Zone is identified when:
      - A strong bearish impulse candle appears (body > 1.5x ATR)
      - Preceded by consolidation

    Returns df with added columns:
      demand_zone_top, demand_zone_bottom, demand_zone_strength,
      demand_zone_fresh, demand_zone_touches,
      supply_zone_top, supply_zone_bottom, supply_zone_strength,
      supply_zone_fresh, supply_zone_touches
    """
    df = df.copy().reset_index(drop=True)
    atr = _atr(df, atr_period)

    # Initialize zone columns
    # Initialize ALL zone columns as float (np.nan) to avoid dtype conflicts
    # when writing float prices (e.g. 21204.08) into columns pandas inferred as int
    for col in [
        "demand_zone_top", "demand_zone_bottom", "demand_zone_strength",
        "demand_zone_fresh", "demand_zone_touches",
        "supply_zone_top", "supply_zone_bottom", "supply_zone_strength",
        "supply_zone_fresh", "supply_zone_touches",
        "nearest_demand_dist_atr", "nearest_supply_dist_atr",
        "in_demand_zone", "in_supply_zone",
        "between_zones",
    ]:
        df[col] = np.nan

    # Rolling zone state
    active_demand = None  # (top, bottom, strength, touches, bar_index)
    active_supply = None

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback: i + 1]
        cur = df.iloc[i]
        cur_atr = atr.iloc[i]

        body = abs(cur["close"] - cur["open"])
        is_bullish = cur["close"] > cur["open"]
        is_bearish = cur["close"] < cur["open"]

        # --- Demand zone detection ---
        if is_bullish and body > impulse_atr_multiplier * cur_atr:
            # Base is the last consolidation candle before this impulse
            base_low = cur["low"]
            base_high = max(cur["open"], cur["close"]) * 0.5 + cur["low"] * 0.5

            strength = min(body / cur_atr, 5.0)  # cap at 5
            active_demand = {
                "top": cur["open"] if cur["close"] > cur["open"] else cur["close"],
                "bottom": cur["low"],
                "strength": strength,
                "touches": 0,
                "bar": i,
                "fresh": True,
            }

        # --- Supply zone detection ---
        if is_bearish and body > impulse_atr_multiplier * cur_atr:
            strength = min(body / cur_atr, 5.0)
            active_supply = {
                "top": cur["high"],
                "bottom": cur["open"] if cur["close"] < cur["open"] else cur["close"],
                "strength": strength,
                "touches": 0,
                "bar": i,
                "fresh": True,
            }

        # --- Write active zone values to row ---
        if active_demand:
            df.at[i, "demand_zone_top"] = active_demand["top"]
            df.at[i, "demand_zone_bottom"] = active_demand["bottom"]
            df.at[i, "demand_zone_strength"] = active_demand["strength"]
            df.at[i, "demand_zone_fresh"] = int(active_demand["fresh"])
            df.at[i, "demand_zone_touches"] = active_demand["touches"]

            dist = (cur["close"] - active_demand["top"]) / cur_atr if cur_atr > 0 else 0
            df.at[i, "nearest_demand_dist_atr"] = dist

            # Check if price is in the demand zone
            in_demand = active_demand["bottom"] <= cur["low"] <= active_demand["top"]
            df.at[i, "in_demand_zone"] = int(in_demand)
            if in_demand:
                active_demand["touches"] += 1
                active_demand["fresh"] = False

        if active_supply:
            df.at[i, "supply_zone_top"] = active_supply["top"]
            df.at[i, "supply_zone_bottom"] = active_supply["bottom"]
            df.at[i, "supply_zone_strength"] = active_supply["strength"]
            df.at[i, "supply_zone_fresh"] = int(active_supply["fresh"])
            df.at[i, "supply_zone_touches"] = active_supply["touches"]

            dist = (active_supply["bottom"] - cur["close"]) / cur_atr if cur_atr > 0 else 0
            df.at[i, "nearest_supply_dist_atr"] = dist

            in_supply = active_supply["bottom"] <= cur["high"] <= active_supply["top"]
            df.at[i, "in_supply_zone"] = int(in_supply)
            if in_supply:
                active_supply["touches"] += 1
                active_supply["fresh"] = False

        # Between zones (no-man's land — avoid trading here)
        if active_demand and active_supply:
            mid_point = (active_supply["bottom"] + active_demand["top"]) / 2
            between = active_demand["top"] < cur["close"] < active_supply["bottom"]
            df.at[i, "between_zones"] = int(between)

    return df


# ---------------------------------------------------------------------------
# Confirmation Signals
# ---------------------------------------------------------------------------

def add_confirmation_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick confirmation signals used in the Zone-to-Zone strategy:

    For buys (demand zone):
      - bullish_engulfing: current candle engulfs prior bearish candle
      - pin_bar_bullish: long lower wick, small body near top
      - higher_low: current low > previous low (momentum shift)
      - bos_bullish: break of structure upward (close above recent swing high)

    For sells (supply zone):
      - bearish_engulfing
      - pin_bar_bearish: long upper wick
      - lower_high: current high < previous high
      - bos_bearish: close below recent swing low
    """
    df = df.copy()

    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    wick_upper = df["high"] - df[["open", "close"]].max(axis=1)
    wick_lower = df[["open", "close"]].min(axis=1) - df["low"]

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_body = (prev_close - prev_open).abs()

    # Bullish engulfing: prev candle bearish, current bullish, current body > prev body
    df["bullish_engulfing"] = (
        (prev_close < prev_open) &          # prev bearish
        (df["close"] > df["open"]) &         # current bullish
        (df["open"] < prev_close) &           # opens below prev close
        (df["close"] > prev_open)             # closes above prev open
    ).astype(int)

    # Bearish engulfing
    df["bearish_engulfing"] = (
        (prev_close > prev_open) &
        (df["close"] < df["open"]) &
        (df["open"] > prev_close) &
        (df["close"] < prev_open)
    ).astype(int)

    # Pin bar bullish: wick_lower > 2x body, wick_lower > wick_upper * 2
    df["pin_bar_bullish"] = (
        (wick_lower > 2 * body.clip(lower=0.0001)) &
        (wick_lower > 2 * wick_upper)
    ).astype(int)

    # Pin bar bearish
    df["pin_bar_bearish"] = (
        (wick_upper > 2 * body.clip(lower=0.0001)) &
        (wick_upper > 2 * wick_lower)
    ).astype(int)

    # Higher low (bullish momentum)
    df["higher_low"] = (df["low"] > df["low"].shift(1)).astype(int)

    # Lower high (bearish momentum)
    df["lower_high"] = (df["high"] < df["high"].shift(1)).astype(int)

    # BOS bullish: close above 5-bar swing high
    swing_high = df["high"].rolling(5).max().shift(1)
    df["bos_bullish"] = (df["close"] > swing_high).astype(int)

    # BOS bearish: close below 5-bar swing low
    swing_low = df["low"].rolling(5).min().shift(1)
    df["bos_bearish"] = (df["close"] < swing_low).astype(int)

    # Composite confirmation scores (0–4)
    df["buy_confirmation_score"] = (
        df["bullish_engulfing"] +
        df["pin_bar_bullish"] +
        df["higher_low"] +
        df["bos_bullish"]
    )

    df["sell_confirmation_score"] = (
        df["bearish_engulfing"] +
        df["pin_bar_bearish"] +
        df["lower_high"] +
        df["bos_bearish"]
    )

    return df


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators relevant to supply/demand zone trading:
      - ATR(14): zone sizing and SL reference
      - RSI(14): overbought/oversold context
      - EMA(20), EMA(50): trend bias
      - EMA spread: fast vs slow distance (normalized by ATR)
      - Bollinger Band position: price position within bands
      - Volume momentum: current vol vs 20-bar average
      - Candle momentum: body/ATR ratio
    """
    df = df.copy()

    # ATR
    df["atr_14"] = _atr(df, 14)

    # RSI
    df["rsi_14"] = _rsi(df["close"], 14)

    # EMAs
    df["ema_20"] = _ema(df["close"], 20)
    df["ema_50"] = _ema(df["close"], 50)
    df["ema_200"] = _ema(df["close"], 200)

    # EMA spread normalized by ATR (trend strength)
    df["ema_spread_atr"] = (df["ema_20"] - df["ema_50"]) / df["atr_14"].replace(0, np.nan)

    # Price position relative to EMAs
    df["price_above_ema20"] = (df["close"] > df["ema_20"]).astype(int)
    df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)
    df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(int)

    # EMA trend bias: +1 bullish, -1 bearish, 0 mixed
    df["ema_trend_bias"] = np.where(
        (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]), 1,
        np.where(
            (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]), -1, 0
        )
    )

    # Bollinger Bands (20, 2)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    df["bb_position"] = (df["close"] - bb_lower) / bb_width.replace(0, np.nan)  # 0–1
    df["bb_width_atr"] = bb_width / df["atr_14"].replace(0, np.nan)

    # Volume momentum
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)

    # Candle body as % of ATR
    body = (df["close"] - df["open"]).abs()
    df["body_atr_ratio"] = body / df["atr_14"].replace(0, np.nan)

    # Momentum: close change over N bars (normalized by ATR)
    df["momentum_5"] = (df["close"] - df["close"].shift(5)) / df["atr_14"].replace(0, np.nan)
    df["momentum_10"] = (df["close"] - df["close"].shift(10)) / df["atr_14"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# HTF Context Alignment
# ---------------------------------------------------------------------------

def add_htf_context(
    ltf_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge higher timeframe zone bias onto lower timeframe rows.

    Strategy rule: Use HTF (1H, 4H) for directional bias, LTF for entry.

    Adds columns:
      htf_1h_bias:   1=bullish zone, -1=bearish zone, 0=neutral
      htf_4h_bias:   same
      htf_aligned:   1 if 1H and 4H agree on direction
    """
    ltf_df = ltf_df.copy()

    def compute_htf_bias(df: pd.DataFrame) -> pd.DataFrame:
        """Compute a simple zone bias per bar."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        atr = _atr(df, 14)
        body = (df["close"] - df["open"]).abs()
        impulse = body > 1.5 * atr

        # Bullish impulse = demand zone = bullish bias
        # Bearish impulse = supply zone = bearish bias
        df["htf_bias"] = np.where(
            impulse & (df["close"] > df["open"]), 1,
            np.where(impulse & (df["close"] < df["open"]), -1, 0)
        )
        # Forward fill bias so all subsequent bars know the last zone type
        df["htf_bias"] = df["htf_bias"].replace(0, np.nan).ffill().fillna(0).astype(int)
        return df[["timestamp", "htf_bias"]]

    ltf_df["timestamp"] = pd.to_datetime(ltf_df["timestamp"])

    if h1_df is not None and len(h1_df) > 0:
        h1_bias = compute_htf_bias(h1_df).rename(columns={"htf_bias": "htf_1h_bias"})
        h1_bias["timestamp"] = pd.to_datetime(h1_bias["timestamp"])
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h1_bias.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
    else:
        ltf_df["htf_1h_bias"] = 0

    if h4_df is not None and len(h4_df) > 0:
        h4_bias = compute_htf_bias(h4_df).rename(columns={"htf_bias": "htf_4h_bias"})
        h4_bias["timestamp"] = pd.to_datetime(h4_bias["timestamp"])
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h4_bias.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
    else:
        ltf_df["htf_4h_bias"] = 0

    # Are both HTFs aligned?
    ltf_df["htf_aligned"] = (
        (ltf_df["htf_1h_bias"] == ltf_df["htf_4h_bias"]) &
        (ltf_df["htf_1h_bias"] != 0)
    ).astype(int)

    return ltf_df


# ---------------------------------------------------------------------------
# Master Feature Builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
    zone_lookback: int = 50,
    impulse_atr_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Full feature pipeline for one timeframe.

    Args:
        df:                      Raw OHLCV DataFrame for the target timeframe
        h1_df:                   1H OHLCV DataFrame for HTF bias
        h4_df:                   4H OHLCV DataFrame for HTF bias
        zone_lookback:           Bars to look back when detecting zones
        impulse_atr_multiplier:  How strong a candle must be (x ATR) to create a zone.
                                 Lower = more zones detected. Default 1.0.

    Returns:
        DataFrame with all features added, NaN rows from warmup dropped.
    """
    logger.info(f"Building features for {len(df)} rows...")

    df = detect_zones(df, lookback=zone_lookback, impulse_atr_multiplier=impulse_atr_multiplier)
    df = add_confirmation_signals(df)
    df = add_indicators(df)

    if h1_df is not None or h4_df is not None:
        df = add_htf_context(df, h1_df, h4_df)

    # Drop warmup rows (EMA 200 needs 200 bars)
    warmup = max(200, zone_lookback)
    df = df.iloc[warmup:].reset_index(drop=True)

    logger.info(f"Features built. Output shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature column list for ML export
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Zone features
    "demand_zone_strength", "demand_zone_fresh", "demand_zone_touches",
    "supply_zone_strength", "supply_zone_fresh", "supply_zone_touches",
    "nearest_demand_dist_atr", "nearest_supply_dist_atr",
    "in_demand_zone", "in_supply_zone", "between_zones",

    # Confirmation signals
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
    "volume_ratio",
    "body_atr_ratio",
    "momentum_5", "momentum_10",

    # HTF context
    "htf_1h_bias", "htf_4h_bias", "htf_aligned",

    # Raw candle context
    "hour", "day_of_week", "month", "session",
    "direction", "candle_size", "body_size", "wick_upper", "wick_lower",
]