"""
labels.py — Zone-to-Zone Label Generation
==========================================
Labels each bar based on Zone-to-Zone rules.

Key design decision:
  Confirmation signals are FEATURES, not gates.
  Any bar inside a zone with valid RR gets labeled.
  The model learns which confirmation patterns matter.

Label values:
   1  = buy signal that hit TP
  -1  = sell signal that hit TP
   0  = no signal, or signal that hit SL / expired
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.labels")


def generate_labels(
    df: pd.DataFrame,
    max_bars: int = 50,
    min_rr: float = 1.0,
    sl_atr_multiplier: float = 1.0,
    use_midline_tp: bool = True,
) -> pd.DataFrame:
    """
    Generate trade labels for each bar.

    Args:
        df:                 Feature-enriched DataFrame from features.py
        max_bars:           Forward bars to simulate TP/SL outcome
        min_rr:             Minimum risk/reward to label as signal (1.0 = break-even+)
        sl_atr_multiplier:  SL distance = N * ATR beyond zone boundary
        use_midline_tp:     TP = midpoint to next zone (safer, per strategy doc)

    Returns:
        df with added columns: signal, trade_outcome, label, tp_price, sl_price, rr_ratio
    """
    df = df.copy().reset_index(drop=True)

    df["signal"]       = 0
    df["signal_reason"]= ""
    df["trade_outcome"]= 0
    df["label"]        = 0
    df["tp_price"]     = np.nan
    df["sl_price"]     = np.nan
    df["rr_ratio"]     = np.nan

    n = len(df)

    for i in range(n - max_bars):
        row = df.iloc[i]

        # Skip no-man's land
        if float(row.get("between_zones", 0) or 0) == 1.0:
            continue

        atr = float(row.get("atr_14", 0) or 0)
        if atr <= 0 or np.isnan(atr):
            continue

        close = float(row["close"])
        signal = 0
        tp = np.nan
        sl = np.nan
        reason = ""

        # ----------------------------------------------------------------
        # BUY — price in demand zone
        # ----------------------------------------------------------------
        if float(row.get("in_demand_zone", 0) or 0) == 1.0:
            demand_bottom = float(row.get("demand_zone_bottom") or np.nan)
            supply_bottom = float(row.get("supply_zone_bottom") or np.nan)

            if np.isnan(demand_bottom):
                continue

            sl = demand_bottom - sl_atr_multiplier * atr

            if not np.isnan(supply_bottom) and use_midline_tp:
                tp = close + (supply_bottom - close) * 0.5
            elif not np.isnan(supply_bottom):
                tp = supply_bottom
            else:
                tp = close + 2.0 * atr  # fallback TP

            risk   = close - sl
            reward = tp - close

            if risk <= 0 or reward <= 0:
                continue

            rr = reward / risk
            if rr < min_rr:
                continue

            signal = 1
            reason = f"demand rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        # ----------------------------------------------------------------
        # SELL — price in supply zone
        # ----------------------------------------------------------------
        elif float(row.get("in_supply_zone", 0) or 0) == 1.0:
            supply_top  = float(row.get("supply_zone_top") or np.nan)
            demand_top  = float(row.get("demand_zone_top") or np.nan)

            if np.isnan(supply_top):
                continue

            sl = supply_top + sl_atr_multiplier * atr

            if not np.isnan(demand_top) and use_midline_tp:
                tp = close - (close - demand_top) * 0.5
            elif not np.isnan(demand_top):
                tp = demand_top
            else:
                tp = close - 2.0 * atr

            risk   = sl - close
            reward = close - tp

            if risk <= 0 or reward <= 0:
                continue

            rr = reward / risk
            if rr < min_rr:
                continue

            signal = -1
            reason = f"supply rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        if signal == 0:
            continue

        df.at[i, "signal"]        = signal
        df.at[i, "signal_reason"] = reason
        df.at[i, "tp_price"]      = float(tp)
        df.at[i, "sl_price"]      = float(sl)

        # ----------------------------------------------------------------
        # Simulate forward price action
        # ----------------------------------------------------------------
        outcome = 0
        future  = df.iloc[i + 1: i + 1 + max_bars]

        for _, fbar in future.iterrows():
            fh = float(fbar["high"])
            fl = float(fbar["low"])
            if signal == 1:
                if fh >= tp:   outcome =  1; break
                if fl <= sl:   outcome = -1; break
            else:
                if fl <= tp:   outcome =  1; break
                if fh >= sl:   outcome = -1; break

        df.at[i, "trade_outcome"] = outcome
        # Winners get directional label; losers get 0
        df.at[i, "label"] = signal if outcome == 1 else 0

    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    signals = (df["signal"] != 0).sum()
    buys    = (df["label"] ==  1).sum()
    sells   = (df["label"] == -1).sum()
    tp_hits = (df["trade_outcome"] == 1).sum()
    sl_hits = (df["trade_outcome"] == -1).sum()
    win_rate = tp_hits / max(signals, 1) * 100

    logger.info(
        f"Labels | signals={signals} buy_wins={buys} sell_wins={sells} "
        f"win_rate={win_rate:.1f}% TP={tp_hits} SL={sl_hits}"
    )


def get_class_weights(df: pd.DataFrame) -> dict:
    """Compute class weights for imbalanced labels."""
    from collections import Counter
    counts = Counter(df["label"])
    total  = len(df)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}