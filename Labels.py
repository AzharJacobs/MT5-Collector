"""
Label Generation Module — Zone-to-Zone Strategy
================================================
Generates ML training labels from OHLCV data based on the Zone-to-Zone rules:

Label logic (per candle):
  BUY  (1):  Price is in a demand zone + at least 1 confirmation signal
             → TP = midline to next supply zone (or full supply zone)
             → SL = below demand zone bottom
             → Label = 1 if TP hit before SL within max_bars

  SELL (-1): Price is in a supply zone + at least 1 confirmation signal
             → TP = midline to next demand zone (or full demand zone)
             → SL = above supply zone top
             → Label = -1 if TP hit before SL within max_bars

  HOLD  (0): Price is between zones (no-man's land), no confirmation,
             or inside a zone with no signal → do nothing

This is a fixed-RR outcome label — we simulate forward price action
to determine if the trade would have worked.

NO lookahead leakage: labels use only data AFTER the signal bar.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger("mt5_collector.labels")


# ---------------------------------------------------------------------------
# Core labeling function
# ---------------------------------------------------------------------------

def generate_labels(
    df: pd.DataFrame,
    max_bars: int = 50,
    min_rr: float = 1.5,
    sl_atr_multiplier: float = 1.0,
    require_confirmation: bool = True,
    use_midline_tp: bool = True,
) -> pd.DataFrame:
    """
    Generate trade labels for each bar based on Zone-to-Zone rules.

    Args:
        df:                    Feature-enriched OHLCV DataFrame (output of features.py)
        max_bars:              Max future bars to simulate trade outcome
        min_rr:                Minimum risk/reward ratio required to label as signal
        sl_atr_multiplier:     SL distance = this * ATR beyond zone boundary
        require_confirmation:  If True, requires at least 1 confirmation signal
        use_midline_tp:        If True, TP = midline between zones (safer, per strategy)

    Returns:
        df with added columns:
          signal:         1 (buy), -1 (sell), 0 (hold)
          signal_reason:  string explaining why signal was generated
          trade_outcome:  1 (TP hit), -1 (SL hit), 0 (expired/no trade)
          label:          final ML label = signal * trade_outcome (1, -1, or 0)
          tp_price:       calculated take profit level
          sl_price:       calculated stop loss level
          rr_ratio:       risk/reward ratio of the setup
    """
    df = df.copy().reset_index(drop=True)

    # Initialize output columns
    df["signal"] = 0
    df["signal_reason"] = ""
    df["trade_outcome"] = 0
    df["label"] = 0
    df["tp_price"] = np.nan
    df["sl_price"] = np.nan
    df["rr_ratio"] = np.nan

    n = len(df)

    for i in range(n - max_bars):
        row = df.iloc[i]

        # Skip no-man's land
        if row.get("between_zones", 0) == 1:
            continue

        atr = row.get("atr_14", 0)
        if atr == 0 or pd.isna(atr):
            continue

        signal = 0
        reason = ""
        tp = np.nan
        sl = np.nan

        # ----------------------------------------------------------------
        # BUY SIGNAL: in demand zone
        # ----------------------------------------------------------------
        if row.get("in_demand_zone", 0) == 1:
            buy_conf = row.get("buy_confirmation_score", 0)

            if not require_confirmation or buy_conf >= 1:
                demand_bottom = row.get("demand_zone_bottom", np.nan)
                supply_bottom = row.get("supply_zone_bottom", np.nan)

                if pd.isna(demand_bottom):
                    continue

                # SL = below demand zone bottom
                sl = demand_bottom - sl_atr_multiplier * atr

                # TP = midline to next supply zone, or supply zone bottom
                if use_midline_tp and not pd.isna(supply_bottom):
                    tp = row["close"] + (supply_bottom - row["close"]) * 0.5
                elif not pd.isna(supply_bottom):
                    tp = supply_bottom
                else:
                    # No visible supply zone — use 2x ATR as fallback TP
                    tp = row["close"] + 2 * atr

                risk = row["close"] - sl
                reward = tp - row["close"]

                if risk <= 0:
                    continue

                rr = reward / risk
                if rr < min_rr:
                    continue

                signal = 1
                reason = f"demand_zone conf={buy_conf} rr={rr:.2f}"
                df.at[i, "rr_ratio"] = rr

        # ----------------------------------------------------------------
        # SELL SIGNAL: in supply zone
        # ----------------------------------------------------------------
        elif row.get("in_supply_zone", 0) == 1:
            sell_conf = row.get("sell_confirmation_score", 0)

            if not require_confirmation or sell_conf >= 1:
                supply_top = row.get("supply_zone_top", np.nan)
                demand_top = row.get("demand_zone_top", np.nan)

                if pd.isna(supply_top):
                    continue

                # SL = above supply zone top
                sl = supply_top + sl_atr_multiplier * atr

                # TP = midline to next demand zone, or demand zone top
                if use_midline_tp and not pd.isna(demand_top):
                    tp = row["close"] - (row["close"] - demand_top) * 0.5
                elif not pd.isna(demand_top):
                    tp = demand_top
                else:
                    tp = row["close"] - 2 * atr

                risk = sl - row["close"]
                reward = row["close"] - tp

                if risk <= 0:
                    continue

                rr = reward / risk
                if rr < min_rr:
                    continue

                signal = -1
                reason = f"supply_zone conf={sell_conf} rr={rr:.2f}"
                df.at[i, "rr_ratio"] = rr

        if signal == 0:
            continue

        # ----------------------------------------------------------------
        # Simulate forward price action to determine outcome
        # ----------------------------------------------------------------
        df.at[i, "signal"] = signal
        df.at[i, "signal_reason"] = reason
        df.at[i, "tp_price"] = tp
        df.at[i, "sl_price"] = sl

        outcome = 0
        future = df.iloc[i + 1: i + 1 + max_bars]

        for _, fbar in future.iterrows():
            if signal == 1:
                if fbar["high"] >= tp:
                    outcome = 1   # TP hit
                    break
                if fbar["low"] <= sl:
                    outcome = -1  # SL hit
                    break
            else:  # signal == -1
                if fbar["low"] <= tp:
                    outcome = 1   # TP hit
                    break
                if fbar["high"] >= sl:
                    outcome = -1  # SL hit
                    break

        df.at[i, "trade_outcome"] = outcome

        # Final label:
        #   signal=1, outcome=1   → label=1  (buy winner)
        #   signal=1, outcome=-1  → label=0  (buy loser — mapped to hold)
        #   signal=-1, outcome=1  → label=-1 (sell winner)
        #   signal=-1, outcome=-1 → label=0  (sell loser)
        #   outcome=0 (expired)   → label=0
        if outcome == 1:
            df.at[i, "label"] = signal      # +1 or -1
        else:
            df.at[i, "label"] = 0           # loser or no trade = hold

    _log_label_summary(df)
    return df


def _log_label_summary(df: pd.DataFrame) -> None:
    total = len(df)
    buys   = (df["label"] == 1).sum()
    sells  = (df["label"] == -1).sum()
    holds  = (df["label"] == 0).sum()
    signals = (df["signal"] != 0).sum()
    tp_hits = (df["trade_outcome"] == 1).sum()
    sl_hits = (df["trade_outcome"] == -1).sum()

    win_rate = tp_hits / signals * 100 if signals > 0 else 0

    logger.info(
        f"Label summary | total={total} | signals={signals} "
        f"| buy_labels={buys} | sell_labels={sells} | hold={holds} "
        f"| win_rate={win_rate:.1f}% (TP={tp_hits}, SL={sl_hits})"
    )

    if signals > 0:
        avg_rr = df.loc[df["signal"] != 0, "rr_ratio"].mean()
        logger.info(f"Average RR ratio on signals: {avg_rr:.2f}")


# ---------------------------------------------------------------------------
# Class imbalance helper
# ---------------------------------------------------------------------------

def get_class_weights(df: pd.DataFrame) -> dict:
    """
    Compute class weights to handle label imbalance.
    Returns dict for use with sklearn: {0: w0, 1: w1, -1: w_neg1}
    """
    from collections import Counter
    counts = Counter(df["label"])
    total = len(df)
    weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
    logger.info(f"Class weights: {weights}")
    return weights