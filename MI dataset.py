"""
ML Dataset Pipeline — Zone-to-Zone Strategy
============================================
Loads raw OHLCV data from PostgreSQL, runs the full feature + label
pipeline, and exports a clean ML-ready dataset.

Usage:
    python ml_dataset.py --timeframe 5min --output dataset_5min.csv
    python ml_dataset.py --timeframe 15min --output dataset_15min.csv
    python ml_dataset.py --all --output_dir ./datasets/
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Add parent dir to path if running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_CONFIG, TIMEFRAMES
from features import build_features, FEATURE_COLUMNS
from labels import generate_labels, get_class_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("mt5_collector.ml_dataset")

# LTF timeframes that need HTF context (1H and 4H injected)
LTF_TIMEFRAMES = ["1min", "2min", "3min", "4min", "5min", "10min", "15min", "30min"]
HTF_TIMEFRAMES = ["1H", "4H", "1D"]


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------

def load_ohlcv(
    timeframe: str,
    symbol: str = "USTECm",
    limit: int = None,
) -> pd.DataFrame:
    """Load OHLCV data for a given timeframe from PostgreSQL."""
    import psycopg2

    query = """
        SELECT
            timestamp, open, high, low, close, volume,
            hour, day_of_week, month, year, session,
            direction, candle_size, body_size, wick_upper, wick_lower
        FROM ustech_ohlcv
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn, params=(symbol, timeframe))
        conn.close()
        logger.info(f"Loaded {len(df)} rows for {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {timeframe}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main pipeline per timeframe
# ---------------------------------------------------------------------------

def build_ml_dataset(
    timeframe: str,
    symbol: str = "USTECm",
    max_label_bars: int = 50,
    min_rr: float = 1.5,
    require_confirmation: bool = True,
    use_midline_tp: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → features → labels → clean dataset.

    Returns a DataFrame ready for XGBoost/RandomForest training.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Building ML dataset for timeframe: {timeframe}")
    logger.info(f"{'='*60}")

    # Load target timeframe
    df = load_ohlcv(timeframe, symbol)
    if df.empty:
        logger.warning(f"No data found for {timeframe}. Skipping.")
        return pd.DataFrame()

    # Load HTF data for context alignment
    h1_df = None
    h4_df = None

    if timeframe in LTF_TIMEFRAMES:
        logger.info("Loading HTF context (1H, 4H)...")
        h1_df = load_ohlcv("1H", symbol)
        h4_df = load_ohlcv("4H", symbol)

        if h1_df.empty:
            logger.warning("No 1H data found for HTF context.")
            h1_df = None
        if h4_df.empty:
            logger.warning("No 4H data found for HTF context.")
            h4_df = None

    # Build features
    df_features = build_features(df, h1_df=h1_df, h4_df=h4_df)

    # Generate labels
    df_labeled = generate_labels(
        df_features,
        max_bars=max_label_bars,
        min_rr=min_rr,
        require_confirmation=require_confirmation,
        use_midline_tp=use_midline_tp,
    )

    # Select final columns
    meta_cols = ["timestamp", "open", "high", "low", "close", "volume",
                 "signal", "signal_reason", "trade_outcome",
                 "label", "tp_price", "sl_price", "rr_ratio"]

    # Only keep feature columns that exist in the dataframe
    available_features = [c for c in FEATURE_COLUMNS if c in df_labeled.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df_labeled.columns]
    if missing:
        logger.warning(f"Missing feature columns (will be skipped): {missing}")

    final_cols = meta_cols + available_features
    final_cols = [c for c in final_cols if c in df_labeled.columns]

    result = df_labeled[final_cols].copy()

    # Add metadata
    result.insert(0, "timeframe", timeframe)
    result.insert(1, "symbol", symbol)

    # Drop rows with too many NaNs in feature columns
    feature_subset = [c for c in available_features if c in result.columns]
    before = len(result)
    result = result.dropna(subset=feature_subset, how="all")
    after = len(result)
    if before != after:
        logger.info(f"Dropped {before - after} rows with all-NaN features")

    # Log class distribution
    label_dist = result["label"].value_counts().to_dict()
    logger.info(f"Label distribution: {label_dist}")

    # Log signal stats
    signals = result[result["signal"] != 0]
    if len(signals) > 0:
        avg_rr = signals["rr_ratio"].mean()
        win_rate = (result["trade_outcome"] == 1).sum() / max(len(signals), 1) * 100
        logger.info(f"Signals: {len(signals)} | Win rate: {win_rate:.1f}% | Avg RR: {avg_rr:.2f}")

    logger.info(f"Final dataset shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Export the ML dataset to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} rows → {output_path}")


def export_train_test_split(
    df: pd.DataFrame,
    output_dir: str,
    timeframe: str,
    test_ratio: float = 0.2,
) -> None:
    """
    Export train/test split (time-based, not random — critical for time series).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, f"{timeframe}_train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, f"{timeframe}_test.csv"), index=False)

    logger.info(
        f"Train/test split | {timeframe} | "
        f"train={len(train)} ({train['timestamp'].min()} → {train['timestamp'].max()}) | "
        f"test={len(test)} ({test['timestamp'].min()} → {test['timestamp'].max()})"
    )


def print_dataset_summary(df: pd.DataFrame, timeframe: str) -> None:
    """Print a human-readable summary of the dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset Summary: {timeframe}")
    print(f"{'='*60}")
    print(f"  Rows:            {len(df):,}")
    print(f"  Date range:      {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Features:        {len([c for c in df.columns if c in FEATURE_COLUMNS])}")

    print(f"\n  Label distribution:")
    for lbl, count in sorted(df["label"].value_counts().items()):
        pct = count / len(df) * 100
        name = {1: "BUY (winner)", -1: "SELL (winner)", 0: "HOLD/loser"}[lbl]
        print(f"    {name:20s}: {count:6,} ({pct:.1f}%)")

    signals = df[df["signal"] != 0]
    if len(signals) > 0:
        print(f"\n  Total signals:   {len(signals):,}")
        tp_hits = (signals["trade_outcome"] == 1).sum()
        sl_hits = (signals["trade_outcome"] == -1).sum()
        expired = (signals["trade_outcome"] == 0).sum()
        print(f"  TP hit:          {tp_hits:,} ({tp_hits/len(signals)*100:.1f}%)")
        print(f"  SL hit:          {sl_hits:,} ({sl_hits/len(signals)*100:.1f}%)")
        print(f"  Expired:         {expired:,} ({expired/len(signals)*100:.1f}%)")
        print(f"  Avg RR ratio:    {signals['rr_ratio'].mean():.2f}")

    # Zone stats
    print(f"\n  Zone encounters:")
    print(f"    In demand zone:  {df['in_demand_zone'].sum():,}")
    print(f"    In supply zone:  {df['in_supply_zone'].sum():,}")
    print(f"    Between zones:   {df['between_zones'].sum():,}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zone-to-Zone ML Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset for 5min timeframe
  python ml_dataset.py --timeframe 5min --output ./datasets/5min.csv

  # Build and split into train/test
  python ml_dataset.py --timeframe 15min --split --output_dir ./datasets/

  # Build all LTF timeframes
  python ml_dataset.py --all --output_dir ./datasets/

  # Adjust label parameters
  python ml_dataset.py --timeframe 5min --min_rr 2.0 --max_bars 30 --output ./datasets/5min.csv
"""
    )

    parser.add_argument("--timeframe", type=str, help="Single timeframe to process")
    parser.add_argument("--all", action="store_true", help="Process all LTF timeframes")
    parser.add_argument("--symbol", type=str, default="USTECm", help="Symbol")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--split", action="store_true", help="Export train/test split")
    parser.add_argument("--min_rr", type=float, default=1.5, help="Min RR ratio for signal")
    parser.add_argument("--max_bars", type=int, default=50, help="Max bars to simulate trade")
    parser.add_argument("--no_confirmation", action="store_true", help="Disable confirmation filter")
    parser.add_argument("--full_tp", action="store_true", help="Use full zone as TP (not midline)")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")

    args = parser.parse_args()

    timeframes_to_run = []
    if args.all:
        timeframes_to_run = LTF_TIMEFRAMES
    elif args.timeframe:
        timeframes_to_run = [args.timeframe]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}

    for tf in timeframes_to_run:
        df = build_ml_dataset(
            timeframe=tf,
            symbol=args.symbol,
            max_label_bars=args.max_bars,
            min_rr=args.min_rr,
            require_confirmation=not args.no_confirmation,
            use_midline_tp=not args.full_tp,
        )

        if df.empty:
            continue

        results[tf] = df

        if args.summary:
            print_dataset_summary(df, tf)

        if args.split:
            export_train_test_split(df, args.output_dir, tf)
        elif args.output:
            export_dataset(df, args.output)
        else:
            out_path = os.path.join(args.output_dir, f"{tf}_ml_dataset.csv")
            export_dataset(df, out_path)

    # Combined summary
    print(f"\n{'='*60}")
    print("All timeframes complete")
    print(f"{'='*60}")
    for tf, df in results.items():
        signals = (df["signal"] != 0).sum()
        labels = (df["label"] != 0).sum()
        print(f"  {tf:8s}: {len(df):>8,} rows | {signals:>5,} signals | {labels:>5,} labeled")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()