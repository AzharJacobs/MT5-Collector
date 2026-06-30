"""
XAUUSD Price Collector
Connects to MetaTrader 5, fetches raw OHLCV data in chunks, and saves it to the DB.

Mirrors collectors/price_collector.py exactly.
Only symbol and database/table references differ (XAUUSD / xauusd_ohlcv).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import time

from xauusd_config import (
    MT5_CONFIG,
    SYMBOL,
    CHUNK_SIZE,
    TIMEFRAMES,
    DATA_START_DATE,
    DATA_END_DATE,
)
from storage.xauusd_db_writer import DatabaseManager
from logger import get_logger, CollectionLogger, setup_logging
from validators import DataValidator, BatchValidationResult
from validators.gap_checker import GapChecker, EXPECTED_CANDLES_PER_DAY

setup_logging()
logger = get_logger("xauusd_collector")

BROKER_UTC_OFFSET = 3

WINDOW_DAYS = {
    "15min": 60,
    "1H":    180,
    "4H":    365,
    "1D":    500,
}


def get_session(timestamp: pd.Timestamp, broker_utc_offset: int = BROKER_UTC_OFFSET) -> str:
    utc_hour = (timestamp.hour - broker_utc_offset) % 24
    if 0 <= utc_hour < 7:
        return "asian"
    elif 7 <= utc_hour < 12:
        return "london"
    elif 12 <= utc_hour < 16:
        return "london_ny_overlap"
    elif 16 <= utc_hour < 21:
        return "new_york"
    else:
        return "off_hours"


def parse_date(date_str, default: datetime) -> datetime:
    if isinstance(date_str, datetime):
        return date_str
    try:
        return datetime.fromisoformat(str(date_str))
    except Exception:
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            logger.warning(f"Could not parse date '{date_str}', using default {default}")
            return default


class MT5Collector:

    TIMEFRAME_MAP = {
        "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
        "TIMEFRAME_H1":  mt5.TIMEFRAME_H1,
        "TIMEFRAME_H4":  mt5.TIMEFRAME_H4,
        "TIMEFRAME_D1":  mt5.TIMEFRAME_D1,
    }

    DAYS_OF_WEEK = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
    }

    def __init__(
        self,
        symbol: str = SYMBOL,
        chunk_size: int = CHUNK_SIZE,
        enable_validation: bool = True,
        broker_utc_offset: int = BROKER_UTC_OFFSET,
    ):
        self.symbol = symbol
        self.chunk_size = chunk_size
        self.enable_validation = enable_validation
        self.broker_utc_offset = broker_utc_offset
        self.db = DatabaseManager()
        self.initialized = False
        self.collection_logger = CollectionLogger()

        self.data_start_date = parse_date(DATA_START_DATE, datetime(2024, 1, 1))
        self.data_end_date   = parse_date(DATA_END_DATE,   datetime(2026, 1, 1))
        self.data_end_date   = self.data_end_date.replace(hour=23, minute=59, second=59)

        self.validator = DataValidator(check_outliers=True, outlier_std_threshold=5.0) \
            if self.enable_validation else None

        self._gap_checker = GapChecker(self.db.get_cursor, symbol, table_name="xauusd_ohlcv")

        logger.info(
            f"MT5Collector ready | symbol={symbol} | "
            f"range={self.data_start_date.date()} → {self.data_end_date.date()} | "
            f"broker_offset=GMT+{broker_utc_offset}"
        )

    # -----------------------------------------------------------------------
    # MT5 connection
    # -----------------------------------------------------------------------
    def initialize(self) -> bool:
        logger.info("Connecting to MT5...")
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        if MT5_CONFIG.get("login"):
            ok = mt5.login(
                MT5_CONFIG["login"],
                password=MT5_CONFIG.get("password", ""),
                server=MT5_CONFIG.get("server", ""),
            )
            if not ok:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            logger.info(f"Logged in: account {MT5_CONFIG['login']}")

        self.initialized = True
        info = mt5.terminal_info()
        if info:
            logger.info(f"Terminal: {info.name} build {info.build}")
        return True

    def shutdown(self):
        mt5.shutdown()
        self.initialized = False
        logger.info("MT5 disconnected")

    def check_symbol(self) -> bool:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error(f"Symbol '{self.symbol}' not found")
            similar = [s.name for s in (mt5.symbols_get() or [])
                       if "XAU" in s.name.upper() or "GOLD" in s.name.upper() or "USD" in s.name.upper()][:8]
            if similar:
                logger.info(f"Possible alternatives: {similar}")
            return False

        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Could not select symbol '{self.symbol}'")
                return False

        logger.info(f"Symbol '{self.symbol}' OK | digits={info.digits} spread={info.spread}")
        return True

    # -----------------------------------------------------------------------
    # Candle builder
    # -----------------------------------------------------------------------
    def _build_candles(self, df: pd.DataFrame, timeframe_name: str) -> List[Dict[str, Any]]:
        candles = []
        for _, row in df.iterrows():
            ts      = pd.to_datetime(row["time"], unit="s")
            open_p  = float(row["open"])
            high_p  = float(row["high"])
            low_p   = float(row["low"])
            close_p = float(row["close"])
            volume  = float(row["tick_volume"])
            spread  = int(row["spread"]) if "spread" in row and not pd.isna(row["spread"]) else 0

            direction = "buy" if close_p > open_p else ("sell" if close_p < open_p else "neutral")
            session   = "daily" if timeframe_name == "1D" else get_session(ts, self.broker_utc_offset)

            candles.append({
                "symbol":      self.symbol,
                "timeframe":   timeframe_name,
                "timestamp":   ts,
                "date":        ts.date(),
                "time":        ts.time(),
                "hour":        ts.hour,
                "day_of_week": self.DAYS_OF_WEEK[ts.weekday()],
                "month":       ts.month,
                "year":        ts.year,
                "open":        open_p,
                "high":        high_p,
                "low":         low_p,
                "close":       close_p,
                "volume":      volume,
                "spread":      spread,
                "direction":   direction,
                "candle_size": high_p - low_p,
                "body_size":   abs(close_p - open_p),
                "wick_upper":  high_p - max(open_p, close_p),
                "wick_lower":  min(open_p, close_p) - low_p,
                "session":     session,
            })
        return candles

    def _validate(self, candles, timeframe) -> Tuple[List, int]:
        if not self.enable_validation or self.validator is None:
            return candles, 0
        result = self.validator.validate_batch(candles)
        if result.invalid_count > 0:
            logger.warning(f"{timeframe}: {result.invalid_count} invalid candles filtered")
        return result.valid_candles, result.invalid_count

    # -----------------------------------------------------------------------
    # Core fetch — forward chunking with overlap
    # -----------------------------------------------------------------------
    def _fetch_timeframe(
        self,
        timeframe_name: str,
        timeframe_const: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[int, int, int]:
        window_days    = WINDOW_DAYS.get(timeframe_name, 30)
        total_fetched  = 0
        total_inserted = 0
        total_invalid  = 0
        session_counts: Dict[str, int] = {}
        current_start = start_date
        chunk_num     = 0

        logger.info(
            f"\n{'─'*50}\n"
            f"  Timeframe : {timeframe_name}\n"
            f"  From      : {start_date.date()}\n"
            f"  To        : {end_date.date()}\n"
            f"  Window    : {window_days} days/chunk\n"
            f"  Est chunks: ~{max(1, (end_date - start_date).days // window_days)}\n"
            f"{'─'*50}"
        )

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=window_days), end_date)
            chunk_num  += 1

            rates = mt5.copy_rates_range(self.symbol, timeframe_const, current_start, current_end)

            if rates is None or len(rates) == 0:
                logger.warning(
                    f"  {timeframe_name} chunk {chunk_num}: NO DATA for "
                    f"{current_start.date()} → {current_end.date()}"
                )
                current_start = current_end
                continue

            df      = pd.DataFrame(rates)
            fetched = len(df)
            total_fetched += fetched

            candles = self._build_candles(df, timeframe_name)
            for c in candles:
                s = c["session"]
                session_counts[s] = session_counts.get(s, 0) + 1

            valid_candles, invalid = self._validate(candles, timeframe_name)
            total_invalid += invalid

            inserted = self.db.insert_candles(valid_candles) if valid_candles else 0
            total_inserted += inserted

            first_ts = pd.to_datetime(df["time"].iloc[0],  unit="s").date()
            last_ts  = pd.to_datetime(df["time"].iloc[-1], unit="s").date()
            logger.info(
                f"  {timeframe_name} chunk {chunk_num:>3}: "
                f"{first_ts} → {last_ts} | "
                f"fetched={fetched:>6,} inserted={inserted:>6,} invalid={invalid}"
            )

            current_start = current_end
            time.sleep(0.05)

        if session_counts:
            summary = ", ".join(f"{s}={n:,}" for s, n in sorted(session_counts.items()))
            logger.info(f"  {timeframe_name} sessions: {summary}")

        logger.info(
            f"  {timeframe_name} COMPLETE | "
            f"fetched={total_fetched:,} inserted={total_inserted:,} invalid={total_invalid}"
        )
        return total_fetched, total_inserted, total_invalid

    # -----------------------------------------------------------------------
    # Gap detection and refetch (delegates to GapChecker)
    # -----------------------------------------------------------------------
    def find_gaps(self, timeframe_name: str):
        return self._gap_checker.find_gaps(timeframe_name)

    def refetch_gaps(self, timeframe_name: str, timeframe_const: int) -> Tuple[int, int]:
        windows = self._gap_checker.get_refetch_windows(timeframe_name)
        if not windows:
            return 0, 0

        total_fetched  = 0
        total_inserted = 0
        for window_start, window_end in windows:
            f, i, _ = self._fetch_timeframe(timeframe_name, timeframe_const, window_start, window_end)
            total_fetched  += f
            total_inserted += i

        logger.info(
            f"  {timeframe_name} gap refetch complete | "
            f"fetched={total_fetched:,} inserted={total_inserted:,}"
        )
        return total_fetched, total_inserted

    # -----------------------------------------------------------------------
    # Incremental fetch
    # -----------------------------------------------------------------------
    def _fetch_incremental(
        self,
        timeframe_name: str,
        timeframe_const: int,
    ) -> Tuple[int, int, int]:
        earliest = self.db.get_earliest_timestamp(self.symbol, timeframe_name)
        latest   = self.db.get_latest_timestamp(self.symbol, timeframe_name)

        if latest is None:
            logger.info(f"  {timeframe_name}: no existing data, full fetch from {self.data_start_date.date()}")
            return self._fetch_timeframe(timeframe_name, timeframe_const, self.data_start_date, self.data_end_date)

        total_fetched = total_inserted = total_invalid = 0

        # Backfill: data exists but starts more than a day after the configured start date.
        # Compare by date only to avoid false triggers when the first candle arrives
        # partway through Jan 1 (e.g. 23:00) while data_start_date is midnight Jan 1.
        if earliest and earliest.date() > (self.data_start_date + timedelta(days=1)).date():
            logger.info(
                f"  {timeframe_name}: historical gap detected — "
                f"DB starts {earliest.date()}, config starts {self.data_start_date.date()}. "
                f"Backfilling {self.data_start_date.date()} → {earliest.date()}"
            )
            f, i, inv = self._fetch_timeframe(
                timeframe_name, timeframe_const, self.data_start_date, earliest
            )
            total_fetched  += f
            total_inserted += i
            total_invalid  += inv

        # Forward fill: fetch from latest up to the configured end date
        if latest >= self.data_end_date:
            logger.info(f"  {timeframe_name}: already up to date (forward)")
        else:
            logger.info(f"  {timeframe_name}: forward fill from {latest.date()} → {self.data_end_date.date()}")
            f, i, inv = self._fetch_timeframe(
                timeframe_name, timeframe_const, latest, self.data_end_date
            )
            total_fetched  += f
            total_inserted += i
            total_invalid  += inv

        return total_fetched, total_inserted, total_invalid

    # -----------------------------------------------------------------------
    # Collect all timeframes
    # -----------------------------------------------------------------------
    def collect_all(self, incremental: bool = True) -> Dict[str, Dict]:
        results = {}

        for timeframe_name, tf_const_name in TIMEFRAMES:
            tf_const = self.TIMEFRAME_MAP[tf_const_name]
            try:
                self.collection_logger.log_timeframe_start(timeframe_name)

                if incremental:
                    fetched, inserted, invalid = self._fetch_incremental(timeframe_name, tf_const)
                else:
                    fetched, inserted, invalid = self._fetch_timeframe(
                        timeframe_name, tf_const, self.data_start_date, self.data_end_date
                    )

                results[timeframe_name] = {"fetched": fetched, "inserted": inserted, "invalid": invalid}
                self.collection_logger.log_timeframe_complete(timeframe_name, fetched, inserted)

            except Exception as e:
                logger.error(f"Error collecting {timeframe_name}: {e}", exc_info=True)
                results[timeframe_name] = {"fetched": 0, "inserted": 0, "invalid": 0, "error": str(e)}

        logger.info("\n" + "="*50)
        logger.info("  Running gap detection on all timeframes...")
        logger.info("="*50)

        for timeframe_name, tf_const_name in TIMEFRAMES:
            tf_const = self.TIMEFRAME_MAP[tf_const_name]
            try:
                self.refetch_gaps(timeframe_name, tf_const)
            except Exception as e:
                logger.error(f"Gap refetch failed for {timeframe_name}: {e}")

        return results

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(self, setup_db: bool = True, incremental: bool = True) -> Dict:
        results = {"success": False, "message": "", "timeframes": {}}
        mode = "incremental" if incremental else "full"
        self.collection_logger.start_collection(self.symbol, mode)

        logger.info(
            f"\n{'='*50}\n"
            f"  Date range confirmed:\n"
            f"  START : {self.data_start_date.date()}\n"
            f"  END   : {self.data_end_date.date()}\n"
            f"{'='*50}"
        )

        try:
            if setup_db:
                logger.info("Setting up database schema...")
                self.db.setup_schema()

            if not self.initialize():
                results["message"] = "MT5 initialization failed"
                return results

            if not self.check_symbol():
                results["message"] = f"Symbol '{self.symbol}' not available"
                return results

            logger.info(
                f"\n{'='*50}\n"
                f"  Starting {'INCREMENTAL' if incremental else 'FULL'} collection\n"
                f"  Symbol : {self.symbol}\n"
                f"  Range  : {self.data_start_date.date()} → {self.data_end_date.date()}\n"
                f"{'='*50}"
            )

            results["timeframes"] = self.collect_all(incremental=incremental)
            results["summary"]    = self.db.get_summary()
            results["success"]    = True
            results["message"]    = "Collection completed successfully"

        except Exception as e:
            results["message"] = f"Collection error: {e}"
            logger.error(results["message"], exc_info=True)
        finally:
            self.shutdown()
            results["stats"] = self.collection_logger.end_collection(results["success"])

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="XAUUSD MT5 OHLCV Data Collector")
    parser.add_argument("--full",           action="store_true")
    parser.add_argument("--skip-db-setup",  action="store_true")
    parser.add_argument("--symbol",         type=str, default=SYMBOL)
    parser.add_argument("--no-validation",  action="store_true")
    parser.add_argument("--broker-offset",  type=int, default=BROKER_UTC_OFFSET)
    parser.add_argument("--check-gaps",     action="store_true")
    args = parser.parse_args()

    collector = MT5Collector(
        symbol=args.symbol,
        enable_validation=not args.no_validation,
        broker_utc_offset=args.broker_offset,
    )

    if args.check_gaps:
        print("Running gap detection only...")
        collector.initialize()
        for timeframe_name, _ in TIMEFRAMES:
            gaps = collector.find_gaps(timeframe_name)
            print(f"  {timeframe_name}: {len(gaps)} gap days")
        collector.shutdown()
        return

    results = collector.run(
        setup_db=not args.skip_db_setup,
        incremental=not args.full,
    )

    print("\n" + "=" * 55)
    if results["success"]:
        print("  Status: SUCCESS")
        if results.get("summary"):
            print("\n  Database totals:")
            for row in results["summary"]:
                print(
                    f"    {row['timeframe']:8s}: {row['total_candles']:>10,} candles | "
                    f"{row['earliest']} → {row['latest']}"
                )
    else:
        print(f"  Status: FAILED — {results['message']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
