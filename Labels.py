"""
labels.py — Zone-to-Zone Label Generation

Changes:
  - timeframe now passed as parameter (was reading from df which was always 'unknown')
  - sl_atr_multiplier auto-set per timeframe (0.5 LTF, 1.0 HTF)
  - require_confirmation actually gates signals when True
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.labels")

LTF_SL_MULTIPLIER = 0.5
HTF_SL_MULTIPLIER = 1.0
LTF_SET = {"1min", "2min", "3min", "4min", "5min", "10min", "15min", "30min"}


def generate_labels(
    df: pd.DataFrame,
    timeframe: str = "unknown",
    max_bars: int = 50,
    min_rr: float = 1.0,
    sl_atr_multiplier: float = None,
    use_midline_tp: bool = True,
    require_confirmation: bool = False,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    if sl_atr_multiplier is None:
        sl_atr_multiplier = LTF_SL_MULTIPLIER if timeframe in LTF_SET else HTF_SL_MULTIPLIER

    logger.info(
        f"Generating labels | timeframe={timeframe} | "
        f"sl_mult={sl_atr_multiplier} | min_rr={min_rr} | "
        f"require_confirmation={require_confirmation}"
    )

    df["signal"]        = 0
    df["signal_reason"] = ""
    df["trade_outcome"] = 0
    df["label"]         = 0
    df["tp_price"]      = np.nan
    df["sl_price"]      = np.nan
    df["rr_ratio"]      = np.nan

    n = len(df)

    for i in range(n - max_bars):
        row = df.iloc[i]

        if float(row.get("between_zones", 0) or 0) == 1.0:
            continue

        atr = float(row.get("atr_14", 0) or 0)
        if atr <= 0 or np.isnan(atr):
            continue

        close  = float(row["close"])
        signal = 0
        tp     = np.nan
        sl     = np.nan
        reason = ""

        # ----------------------------------------------------------------
        # BUY — price in demand zone
        # ----------------------------------------------------------------
        if float(row.get("in_demand_zone", 0) or 0) == 1.0:

            if require_confirmation:
                if float(row.get("buy_confirmation_score", 0) or 0) < 1:
                    continue

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
                tp = close + 2.0 * atr

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

            if require_confirmation:
                if float(row.get("sell_confirmation_score", 0) or 0) < 1:
                    continue

            supply_top = float(row.get("supply_zone_top") or np.nan)
            demand_top = float(row.get("demand_zone_top") or np.nan)

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

        # Simulate forward price action
        outcome = 0
        future  = df.iloc[i + 1: i + 1 + max_bars]

        for _, fbar in future.iterrows():
            fh = float(fbar["high"])
            fl = float(fbar["low"])
            if signal == 1:
                if fh >= tp: outcome =  1; break
                if fl <= sl: outcome = -1; break
            else:
                if fl <= tp: outcome =  1; break
                if fh >= sl: outcome = -1; break

        df.at[i, "trade_outcome"] = outcome
        df.at[i, "label"]         = signal if outcome == 1 else 0

    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    signals  = (df["signal"] != 0).sum()
    buys     = (df["label"] ==  1).sum()
    sells    = (df["label"] == -1).sum()
    tp_hits  = (df["trade_outcome"] == 1).sum()
    sl_hits  = (df["trade_outcome"] == -1).sum()
    win_rate = tp_hits / max(signals, 1) * 100
    logger.info(
        f"Labels | signals={signals} buy_wins={buys} sell_wins={sells} "
        f"win_rate={win_rate:.1f}% TP={tp_hits} SL={sl_hits}"
    )


def get_class_weights(df: pd.DataFrame) -> dict:
    from collections import Counter
    counts = Counter(df["label"])
    total  = len(df)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}