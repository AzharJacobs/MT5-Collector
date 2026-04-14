"""
MT5 Data Collector — Zone-to-Zone ML Build
==========================================
Fetches historical OHLCV data from MetaTrader 5 and stores in PostgreSQL.

Collection strategy:
  - Fetches each timeframe fully (start → end) before moving to the next
  - Uses FORWARD chunking: walks start → end in fixed-day windows
  - Window sizes are calibrated per timeframe to stay under MT5 bar limits
  - Safe to re-run: duplicate candles are silently skipped (ON CONFLICT DO NOTHING)
  - Incremental mode: picks up from the last stored timestamp per timeframe
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time

from config import (
    MT5_CONFIG,
    SYMBOL,
    CHUNK_SIZE,
    TIMEFRAMES,
    DATA_START_DATE,
    DATA_END_DATE,
)
from database import DatabaseManager
from logger import get_logger, CollectionLogger, setup_logging
from validator import DataValidator, BatchValidationResult

setup_logging()
logger = get_logger("mt5_collector")

# ---------------------------------------------------------------------------
# Broker UTC offset
# Most brokers (ICMarkets, Exness, Pepperstone) use GMT+2 (winter) / GMT+3 (DST)
# Exness typically uses GMT+3 — adjust if your candle times look off by 1 hour
# ---------------------------------------------------------------------------
BROKER_UTC_OFFSET = 3  # Exness-MT5Real27 uses GMT+3


# ---------------------------------------------------------------------------
# Window sizes per timeframe (days fetched per chunk)
# Tuned to stay comfortably under MT5's ~100k bar limit per request
# ---------------------------------------------------------------------------
WINDOW_DAYS = {
    "1min":  7,    # ~10,080 bars/week — safe
    "2min":  14,
    "3min":  21,
    "4min":  21,
    "5min":  30,   # ~8,640 bars/month
    "10min": 60,
    "15min": 60,   # ~5,760 bars/month
    "30min": 90,
    "1H":    180,
    "4H":    365,
    "1D":    1825, # 5 years in one shot — very few bars
}


# ---------------------------------------------------------------------------
# Session classifier
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class MT5Collector:

    TIMEFRAME_MAP = {
        "TIMEFRAME_M1":  mt5.TIMEFRAME_M1,
        "TIMEFRAME_M2":  mt5.TIMEFRAME_M2,
        "TIMEFRAME_M3":  mt5.TIMEFRAME_M3,
        "TIMEFRAME_M4":  mt5.TIMEFRAME_M4,
        "TIMEFRAME_M5":  mt5.TIMEFRAME_M5,
        "TIMEFRAME_M10": mt5.TIMEFRAME_M10,
        "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
        "TIMEFRAME_M30": mt5.TIMEFRAME_M30,
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
        self.data_end_date   = parse_date(DATA_END_DATE,   datetime(2025, 1, 1))

        # Always end at midnight of end date so we get complete days
        self.data_end_date = self.data_end_date.replace(hour=23, minute=59, second=59)

        if self.enable_validation:
            self.validator = DataValidator(check_outliers=True, outlier_std_threshold=5.0)
        else:
            self.validator = None

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
            logger.error(f"Symbol '{self.symbol}' not found on this broker")
            similar = [s.name for s in (mt5.symbols_get() or [])
                       if "TECH" in s.name.upper() or "NAS" in s.name.upper() or "US" in s.name.upper()][:8]
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
        """Convert raw MT5 rates DataFrame into candle dicts with derived columns."""
        candles = []
        for _, row in df.iterrows():
            ts          = pd.to_datetime(row["time"], unit="s")
            open_p      = float(row["open"])
            high_p      = float(row["high"])
            low_p       = float(row["low"])
            close_p     = float(row["close"])
            volume      = float(row["tick_volume"])

            direction   = "buy" if close_p > open_p else ("sell" if close_p < open_p else "neutral")
            session     = "daily" if timeframe_name == "1D" else get_session(ts, self.broker_utc_offset)

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
    # Core fetch — FORWARD chunking
    # -----------------------------------------------------------------------
    def _fetch_timeframe(
        self,
        timeframe_name: str,
        timeframe_const: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[int, int, int]:
        """
        Fetch one timeframe from start_date to end_date using forward chunking.

        Walks start → end in fixed-day windows (see WINDOW_DAYS).
        Each chunk calls copy_rates_range(symbol, tf, window_start, window_end).
        Inserts results immediately; ON CONFLICT DO NOTHING handles overlaps.

        Returns (total_fetched, total_inserted, total_invalid)
        """
        window_days = WINDOW_DAYS.get(timeframe_name, 30)
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

            rates = mt5.copy_rates_range(
                self.symbol,
                timeframe_const,
                current_start,
                current_end,
            )

            if rates is None or len(rates) == 0:
                logger.debug(
                    f"  {timeframe_name} chunk {chunk_num}: "
                    f"no data {current_start.date()} → {current_end.date()}"
                )
                current_start = current_end + timedelta(seconds=1)
                continue

            df       = pd.DataFrame(rates)
            fetched  = len(df)
            total_fetched += fetched

            candles = self._build_candles(df, timeframe_name)

            # Track session distribution
            for c in candles:
                s = c["session"]
                session_counts[s] = session_counts.get(s, 0) + 1

            valid_candles, invalid = self._validate(candles, timeframe_name)
            total_invalid += invalid

            inserted = self.db.insert_candles(valid_candles) if valid_candles else 0
            total_inserted += inserted

            # Progress log
            first_ts = pd.to_datetime(df["time"].iloc[0],  unit="s").date()
            last_ts  = pd.to_datetime(df["time"].iloc[-1], unit="s").date()
            logger.info(
                f"  {timeframe_name} chunk {chunk_num:>3}: "
                f"{first_ts} → {last_ts} | "
                f"fetched={fetched:>6,} inserted={inserted:>6,} invalid={invalid}"
            )

            # Advance window — start from the day after last bar to avoid overlap
            current_start = current_end + timedelta(seconds=1)
            time.sleep(0.05)  # gentle rate limiting

        # Session summary
        if session_counts:
            summary = ", ".join(f"{s}={n:,}" for s, n in sorted(session_counts.items()))
            logger.info(f"  {timeframe_name} sessions: {summary}")

        logger.info(
            f"  {timeframe_name} COMPLETE | "
            f"fetched={total_fetched:,} inserted={total_inserted:,} invalid={total_invalid}"
        )

        return total_fetched, total_inserted, total_invalid

    # -----------------------------------------------------------------------
    # Incremental fetch
    # -----------------------------------------------------------------------
    def _fetch_incremental(
        self,
        timeframe_name: str,
        timeframe_const: int,
    ) -> Tuple[int, int, int]:
        """
        Fetch only new candles since the last stored timestamp.
        Falls back to full fetch if no existing data.
        """
        latest = self.db.get_latest_timestamp(self.symbol, timeframe_name)

        if latest:
            if latest >= self.data_end_date:
                logger.info(
                    f"  {timeframe_name}: already up to date "
                    f"(latest={latest.date()}, end={self.data_end_date.date()})"
                )
                return 0, 0, 0
            start = latest + timedelta(seconds=1)
            logger.info(f"  {timeframe_name}: incremental from {start.date()}")
        else:
            start = self.data_start_date
            logger.info(f"  {timeframe_name}: no existing data, full fetch")

        return self._fetch_timeframe(timeframe_name, timeframe_const, start, self.data_end_date)

    # -----------------------------------------------------------------------
    # Collect all timeframes sequentially
    # -----------------------------------------------------------------------
    def collect_all(self, incremental: bool = True) -> Dict[str, Dict]:
        results = {}

        for timeframe_name, tf_const_name in TIMEFRAMES:
            tf_const = self.TIMEFRAME_MAP[tf_const_name]

            try:
                self.collection_logger.log_timeframe_start(timeframe_name)

                if incremental:
                    fetched, inserted, invalid = self._fetch_incremental(
                        timeframe_name, tf_const
                    )
                else:
                    fetched, inserted, invalid = self._fetch_timeframe(
                        timeframe_name, tf_const,
                        self.data_start_date, self.data_end_date,
                    )

                results[timeframe_name] = {
                    "fetched":  fetched,
                    "inserted": inserted,
                    "invalid":  invalid,
                }
                self.collection_logger.log_timeframe_complete(
                    timeframe_name, fetched, inserted
                )

            except Exception as e:
                logger.error(f"Error collecting {timeframe_name}: {e}", exc_info=True)
                results[timeframe_name] = {
                    "fetched": 0, "inserted": 0, "invalid": 0, "error": str(e)
                }

        return results

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(self, setup_db: bool = True, incremental: bool = True) -> Dict:
        results = {"success": False, "message": "", "timeframes": {}}
        mode = "incremental" if incremental else "full"
        self.collection_logger.start_collection(self.symbol, mode)

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

    parser = argparse.ArgumentParser(description="MT5 OHLCV Data Collector")
    parser.add_argument("--full",           action="store_true", help="Full history fetch (ignores existing data)")
    parser.add_argument("--skip-db-setup",  action="store_true", help="Skip DB schema creation")
    parser.add_argument("--symbol",         type=str, default=SYMBOL)
    parser.add_argument("--no-validation",  action="store_true", help="Skip data validation")
    parser.add_argument("--broker-offset",  type=int, default=BROKER_UTC_OFFSET)
    args = parser.parse_args()

    print("=" * 55)
    print("  MT5 OHLCV Collector — Zone-to-Zone ML Build")
    print("=" * 55)
    print(f"  Symbol         : {args.symbol}")
    print(f"  Mode           : {'FULL' if args.full else 'INCREMENTAL'}")
    print(f"  DB setup       : {'skip' if args.skip_db_setup else 'yes'}")
    print(f"  Validation     : {'off' if args.no_validation else 'on'}")
    print(f"  Broker offset  : GMT+{args.broker_offset}")
    print("=" * 55)

    collector = MT5Collector(
        symbol=args.symbol,
        enable_validation=not args.no_validation,
        broker_utc_offset=args.broker_offset,
    )
    results = collector.run(
        setup_db=not args.skip_db_setup,
        incremental=not args.full,
    )

    print("\n" + "=" * 55)
    if results["success"]:
        print("  Status: SUCCESS")
        print("\n  Timeframe results:")
        total_f = total_i = total_inv = 0
        for tf, d in results["timeframes"].items():
            f, i, inv = d.get("fetched",0), d.get("inserted",0), d.get("invalid",0)
            total_f += f; total_i += i; total_inv += inv
            err = f"  ⚠ {d['error']}" if "error" in d else ""
            print(f"    {tf:8s}: fetched={f:>8,}  inserted={i:>8,}  invalid={inv}{err}")
        print(f"    {'TOTAL':8s}: fetched={total_f:>8,}  inserted={total_i:>8,}  invalid={total_inv}")

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
    print("  Logs: ./logs/")
    print("=" * 55)


if __name__ == "__main__":
    main()