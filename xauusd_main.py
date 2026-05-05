"""
xauusd_main.py — XAUUSD Verification Pipeline
Reads raw candles from xauusd_ohlcv, runs gap / duplicate / anomaly checks,
and writes all passing rows to xauusd_verified with all flags set to TRUE.
"""

import sys
import os
import shutil
from typing import List, Dict, Any


def check_env_file() -> bool:
    if not os.path.exists('xauusd.env'):
        if os.path.exists('xauusd.env.example'):
            print("Creating xauusd.env from template...")
            shutil.copy('xauusd.env.example', 'xauusd.env')
            print("Edit xauusd.env with your credentials, then run again.")
            return False
        print("Error: xauusd.env not found.")
        return False
    return True


def _fetch_raw(db, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Pull every row for a symbol+timeframe from the raw xauusd_ohlcv table."""
    query = """
    SELECT id, symbol, timeframe, timestamp, date, time, hour,
           day_of_week, month, year, open, high, low, close,
           volume, spread, direction, candle_size, body_size,
           wick_upper, wick_lower, session
    FROM xauusd_ohlcv
    WHERE symbol = %s AND timeframe = %s
    ORDER BY timestamp ASC;
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (symbol, timeframe))
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def run_verification(symbol: str = None) -> Dict[str, Any]:
    from storage.xauusd_db_writer import DatabaseManager
    from validators.gap_checker import GapChecker
    from validators.duplicate_checker import DuplicateChecker
    from validators.anomaly_detector import AnomalyDetector
    from validators.schema_validator import ValidationResult
    from xauusd_config import SYMBOL, TIMEFRAMES
    from logger import get_logger

    logger = get_logger('xauusd_collector.verify')

    if symbol is None:
        symbol = SYMBOL

    timeframe_names = [tf for tf, _ in TIMEFRAMES]

    db = DatabaseManager()
    db.setup_schema()

    gap_checker = GapChecker(db.get_cursor, symbol, table_name="xauusd_ohlcv")

    total_fetched         = 0
    total_written         = 0
    total_dropped_dups    = 0
    total_dropped_anomaly = 0

    for tf in timeframe_names:
        logger.info("=" * 55)
        logger.info(f"Verifying {symbol} {tf}")

        # 1. Fetch from xauusd_ohlcv
        candles = _fetch_raw(db, symbol, tf)
        if not candles:
            logger.warning(f"  No data for {tf} — skipping")
            continue
        logger.info(f"  Fetched:          {len(candles):>10,}")
        total_fetched += len(candles)

        # 2. Gap check — informational only; logs low-count days, does not drop rows
        gap_checker.find_gaps(tf)

        # 3. Duplicate check — keep first occurrence, discard the rest
        unique, dupes = DuplicateChecker.find_duplicates(candles)
        total_dropped_dups += len(dupes)
        logger.info(f"  Duplicates:       {len(dupes):>10,} dropped")

        # 4. Anomaly check — drop any candle with a hard validation error
        detector = AnomalyDetector()
        passing: List[Dict[str, Any]] = []
        for candle in unique:
            result = ValidationResult(is_valid=True)
            detector.check_ohlcv_logic(candle, result)
            detector.check_price_range(candle, result)
            detector.check_volume(candle, result)
            detector.check_timestamp(candle, result)
            detector.check_outliers(candle, result)
            if result.is_valid:
                passing.append(candle)

        dropped_anomaly = len(unique) - len(passing)
        total_dropped_anomaly += dropped_anomaly
        logger.info(f"  Anomalies:        {dropped_anomaly:>10,} dropped")

        # 5. Write to xauusd_verified with all check flags = TRUE
        inserted = db.insert_verified_candles(passing)
        total_written += inserted
        logger.info(f"  Written:          {inserted:>10,}")

    return {
        "fetched":            total_fetched,
        "written":            total_written,
        "dropped_duplicates": total_dropped_dups,
        "dropped_anomalies":  total_dropped_anomaly,
    }


def main():
    print("=" * 55)
    print("XAUUSD MT5 OHLCV — Verification Pipeline")
    print("=" * 55)

    if not check_env_file():
        sys.exit(1)

    try:
        results = run_verification()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("COMPLETE")
    print(f"  Fetched from xauusd_ohlcv:    {results['fetched']:>10,}")
    print(f"  Dropped  (duplicates):        {results['dropped_duplicates']:>10,}")
    print(f"  Dropped  (anomalies):         {results['dropped_anomalies']:>10,}")
    print(f"  Written to xauusd_verified:   {results['written']:>10,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
