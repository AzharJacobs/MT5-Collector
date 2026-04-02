"""
MT5 Data Collector
Fetches historical OHLCV data from MetaTrader 5 and stores it in PostgreSQL.
Includes trading session labeling (Asian, London, London/NY Overlap, New York).
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time

from config import MT5_CONFIG, SYMBOL, CHUNK_SIZE, TIMEFRAMES
from database import DatabaseManager
from logger import get_logger, CollectionLogger, setup_logging
from validator import DataValidator, BatchValidationResult

# Initialize production logging
setup_logging()
logger = get_logger('mt5_collector')


# ---------------------------------------------------------------------------
# Session Classification
# ---------------------------------------------------------------------------
# All session boundaries are defined in UTC.
# MT5 broker timestamps are typically in broker local time (GMT+2 or GMT+3).
# Set BROKER_UTC_OFFSET to match your broker:
#   - Most brokers (ICMarkets, Pepperstone, etc.) use GMT+2 (winter) / GMT+3 (summer DST)
#   - Check your broker's server time vs UTC and set accordingly.
#   - You can verify with: mt5.terminal_info() after connecting.
#
# Session windows (UTC):
#   Asian              00:00 - 07:00
#   London             07:00 - 12:00
#   London / NY Overlap 12:00 - 16:00
#   New York           16:00 - 21:00
#   Off Hours          21:00 - 00:00
# ---------------------------------------------------------------------------
BROKER_UTC_OFFSET = 2  # Change to 3 if your broker uses GMT+3


def get_session(timestamp: pd.Timestamp, broker_utc_offset: int = BROKER_UTC_OFFSET) -> str:
    """
    Classify a candle timestamp into a trading session.

    Converts broker local time to UTC using the configured offset,
    then maps to one of five session labels:
        - 'asian'              00:00-07:00 UTC
        - 'london'             07:00-12:00 UTC
        - 'london_ny_overlap'  12:00-16:00 UTC  (highest volume window)
        - 'new_york'           16:00-21:00 UTC
        - 'off_hours'          21:00-00:00 UTC

    Args:
        timestamp:         Candle open time in broker local time.
        broker_utc_offset: Hours to subtract to convert broker time → UTC.

    Returns:
        Session label as a string.
    """
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


class MT5Collector:
    """Handles MetaTrader 5 connection and data collection"""

    # MT5 timeframe constants mapping
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

    # Days of week mapping
    DAYS_OF_WEEK = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    def __init__(
        self,
        symbol: str = SYMBOL,
        chunk_size: int = CHUNK_SIZE,
        enable_validation: bool = True,
        broker_utc_offset: int = BROKER_UTC_OFFSET
    ):
        self.symbol = symbol
        self.chunk_size = chunk_size
        self.enable_validation = enable_validation
        self.broker_utc_offset = broker_utc_offset
        self.db = DatabaseManager()
        self.initialized = False
        self.collection_logger = CollectionLogger()

        # Initialize validator if enabled
        if self.enable_validation:
            self.validator = DataValidator(
                check_outliers=True,
                outlier_std_threshold=5.0
            )
        else:
            self.validator = None

        logger.info(
            f"MT5Collector initialized | symbol={symbol} | "
            f"broker_utc_offset=GMT+{broker_utc_offset}"
        )

    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        logger.info("Initializing MT5 connection...")

        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed. Error: {error}")
            return False

        # Optional: Login with credentials if provided
        if MT5_CONFIG.get('login'):
            login_result = mt5.login(
                MT5_CONFIG['login'],
                password=MT5_CONFIG.get('password', ''),
                server=MT5_CONFIG.get('server', '')
            )
            if not login_result:
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}")
                return False
            logger.info(f"Logged in to MT5 account: {MT5_CONFIG['login']}")

        self.initialized = True

        # Log terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logger.info(f"MT5 Terminal: {terminal_info.name}")
            logger.info(f"MT5 Version: {terminal_info.build}")

        # Remind about broker offset
        logger.info(
            f"Session labeling active | broker offset: GMT+{self.broker_utc_offset} | "
            f"Sessions: asian(00-07 UTC), london(07-12 UTC), "
            f"london_ny_overlap(12-16 UTC), new_york(16-21 UTC), off_hours(21-00 UTC)"
        )

        return True

    def shutdown(self) -> None:
        """Shutdown MT5 connection"""
        mt5.shutdown()
        self.initialized = False
        logger.info("MT5 connection closed")

    def check_symbol(self) -> bool:
        """Check if the symbol exists and is available"""
        symbol_info = mt5.symbol_info(self.symbol)

        if symbol_info is None:
            logger.error(f"Symbol '{self.symbol}' not found")
            all_symbols = mt5.symbols_get()
            if all_symbols:
                similar = [s.name for s in all_symbols
                           if 'TECH' in s.name.upper() or 'NAS' in s.name.upper()][:5]
                if similar:
                    logger.info(f"Similar symbols found: {similar}")
            return False

        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select symbol '{self.symbol}'")
                return False

        logger.info(f"Symbol '{self.symbol}' is available")
        logger.debug(f"Symbol digits: {symbol_info.digits}, spread: {symbol_info.spread}")
        return True

    def calculate_derived_columns(
        self,
        df: pd.DataFrame,
        timeframe_name: str
    ) -> List[Dict[str, Any]]:
        """
        Calculate all derived columns from OHLCV data including session label.
        Returns a list of dictionaries ready for database insertion.

        Session is determined by converting the broker timestamp to UTC
        using self.broker_utc_offset, then mapping to a named session window.
        Daily candles (1D) are always labeled 'daily' since they span all sessions.
        """
        candles = []

        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row['time'], unit='s')
            open_price  = float(row['open'])
            high_price  = float(row['high'])
            low_price   = float(row['low'])
            close_price = float(row['close'])
            volume      = float(row['tick_volume'])

            # Direction
            if close_price > open_price:
                direction = "buy"
            elif close_price < open_price:
                direction = "sell"
            else:
                direction = "neutral"

            # Candle metrics
            candle_size = high_price - low_price
            body_size   = abs(close_price - open_price)
            wick_upper  = high_price - max(open_price, close_price)
            wick_lower  = min(open_price, close_price) - low_price

            # Session label
            # Daily candles span all sessions so we mark them separately
            if timeframe_name == "1D":
                session = "daily"
            else:
                session = get_session(timestamp, self.broker_utc_offset)

            candle = {
                'symbol':      self.symbol,
                'timeframe':   timeframe_name,
                'timestamp':   timestamp,
                'date':        timestamp.date(),
                'time':        timestamp.time(),
                'hour':        timestamp.hour,
                'day_of_week': self.DAYS_OF_WEEK[timestamp.weekday()],
                'month':       timestamp.month,
                'year':        timestamp.year,
                'open':        open_price,
                'high':        high_price,
                'low':         low_price,
                'close':       close_price,
                'volume':      volume,
                'direction':   direction,
                'candle_size': candle_size,
                'body_size':   body_size,
                'wick_upper':  wick_upper,
                'wick_lower':  wick_lower,
                'session':     session,
            }
            candles.append(candle)

        return candles

    def validate_and_filter_candles(
        self,
        candles: List[Dict[str, Any]],
        timeframe: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Validate candles and filter out invalid ones.
        Returns (valid_candles, invalid_count).
        """
        if not self.enable_validation or self.validator is None:
            return candles, 0

        result = self.validator.validate_batch(candles)

        if result.warning_count > 0:
            logger.debug(
                f"{timeframe}: {result.warning_count} candles with warnings"
            )

        if result.invalid_count > 0:
            logger.warning(
                f"{timeframe}: {result.invalid_count} invalid candles filtered out"
            )
            for candle, errors in result.invalid_candles[:5]:
                logger.debug(f"  Invalid candle at {candle.get('timestamp')}: {errors}")

        return result.valid_candles, result.invalid_count

    def fetch_historical_data(
        self,
        timeframe_name: str,
        timeframe_const: int,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Tuple[int, int, int]:
        """
        Fetch historical data for a specific timeframe in chunks.
        For lower timeframes, uses sliding window to avoid MT5 bar limits.
        Returns tuple of (total_fetched, total_inserted, total_invalid).
        """
        if not self.initialized:
            raise RuntimeError("MT5 not initialized. Call initialize() first.")

        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = datetime(2000, 1, 1)

        logger.info(f"Fetching {timeframe_name} from {start_date} to {end_date}")
        self.collection_logger.log_timeframe_start(timeframe_name)

        total_fetched    = 0
        total_inserted   = 0
        total_invalid    = 0

        # Session distribution tracker for logging
        session_counts: Dict[str, int] = {}

        # For lower timeframes, use sliding window to avoid MT5 limits
        if timeframe_name in ["1min", "2min", "3min", "4min", "5min", "10min", "15min", "30min"]:
            window_days = 30  # Fetch 30 days at a time
        elif timeframe_name in ["1H"]:
            window_days = 90  # 90 days for 1H
        else:
            window_days = None  # No limit for 4H, 1D

        if window_days:
            # Sliding window approach for lower timeframes
            current_end = end_date
            
            while current_end > start_date:
                current_start = max(start_date, current_end - timedelta(days=window_days))
                
                rates = mt5.copy_rates_range(
                    self.symbol,
                    timeframe_const,
                    current_start,
                    current_end
                )
                
                if rates is None or len(rates) == 0:
                    current_end = current_start - timedelta(seconds=1)
                    continue
                    
                df = pd.DataFrame(rates)
                chunk_size = len(df)
                total_fetched += chunk_size

                # Calculate derived columns (includes session)
                candles = self.calculate_derived_columns(df, timeframe_name)

                # Track session distribution for this chunk
                for c in candles:
                    s = c['session']
                    session_counts[s] = session_counts.get(s, 0) + 1

                # Validate and filter
                valid_candles, invalid_count = self.validate_and_filter_candles(
                    candles, timeframe_name
                )
                total_invalid += invalid_count

                # Insert into database
                inserted = 0
                if valid_candles:
                    inserted = self.db.insert_candles(valid_candles)
                    total_inserted += inserted

                # Earliest timestamp in this chunk
                earliest_time = pd.to_datetime(df['time'].min(), unit='s')

                self.collection_logger.log_chunk_processed(
                    timeframe=timeframe_name,
                    fetched=chunk_size,
                    inserted=inserted,
                    invalid=invalid_count,
                    earliest=str(earliest_time)
                )

                logger.info(
                    f"  {timeframe_name}: Fetched {chunk_size}, "
                    f"inserted {inserted}, invalid {invalid_count} "
                    f"(up to {earliest_time})"
                )

                current_end = current_start - timedelta(seconds=1)
        else:
            # Original logic for higher timeframes (4H, 1D)
            current_end      = end_date
            consecutive_empty = 0
            max_consecutive_empty = 3

            while current_end > start_date and consecutive_empty < max_consecutive_empty:

                rates = mt5.copy_rates_range(
                    self.symbol,
                    timeframe_const,
                    start_date,
                    current_end
                )

                if rates is None or len(rates) == 0:
                    consecutive_empty += 1
                    logger.debug(f"No more data for {timeframe_name} before {current_end}")

                    # Step back and retry
                    if timeframe_name == "1D":
                        current_end -= timedelta(days=365)
                    elif timeframe_name in ["1H", "4H"]:
                        current_end -= timedelta(days=30)
                    else:
                        current_end -= timedelta(days=7)
                    continue

                consecutive_empty = 0

                df = pd.DataFrame(rates)
                chunk_size = len(df)
                total_fetched += chunk_size

                # Calculate derived columns (includes session)
                candles = self.calculate_derived_columns(df, timeframe_name)

                # Track session distribution for this chunk
                for c in candles:
                    s = c['session']
                    session_counts[s] = session_counts.get(s, 0) + 1

                # Validate and filter
                valid_candles, invalid_count = self.validate_and_filter_candles(
                    candles, timeframe_name
                )
                total_invalid += invalid_count

                # Insert into database
                inserted = 0
                if valid_candles:
                    inserted = self.db.insert_candles(valid_candles)
                    total_inserted += inserted

                # Earliest timestamp in this chunk
                earliest_time = pd.to_datetime(df['time'].min(), unit='s')

                self.collection_logger.log_chunk_processed(
                    timeframe=timeframe_name,
                    fetched=chunk_size,
                    inserted=inserted,
                    invalid=invalid_count,
                    earliest=str(earliest_time)
                )

                logger.info(
                    f"  {timeframe_name}: Fetched {chunk_size}, "
                    f"inserted {inserted}, invalid {invalid_count} "
                    f"(up to {earliest_time})"
                )

                # Reached beginning of available data
                if chunk_size < self.chunk_size / 2:
                    break

                current_end = earliest_time - timedelta(seconds=1)
                time.sleep(0.1)

        # Log session distribution summary for this timeframe
        if session_counts:
            session_summary = ", ".join(
                f"{s}={n:,}" for s, n in sorted(session_counts.items())
            )
            logger.info(f"  {timeframe_name} session breakdown: {session_summary}")

        self.collection_logger.log_timeframe_complete(
            timeframe_name, total_fetched, total_inserted
        )

        return total_fetched, total_inserted, total_invalid

    def fetch_incremental_data(
        self,
        timeframe_name: str,
        timeframe_const: int
    ) -> Tuple[int, int, int]:
        """
        Fetch only new data since the last stored timestamp.
        Returns tuple of (total_fetched, total_inserted, total_invalid).
        """
        latest = self.db.get_latest_timestamp(self.symbol, timeframe_name)

        if latest:
            start_date = latest + timedelta(seconds=1)
            logger.info(f"Incremental fetch for {timeframe_name} from {start_date}")
        else:
            start_date = None
            logger.info(f"No existing data for {timeframe_name}, doing full fetch")

        return self.fetch_historical_data(
            timeframe_name,
            timeframe_const,
            start_date=start_date
        )

    def collect_all_timeframes(
        self,
        incremental: bool = True
    ) -> Dict[str, Dict[str, int]]:
        """
        Collect data for all configured timeframes sequentially.
        Returns a summary dictionary per timeframe.
        """
        results = {}

        for timeframe_name, timeframe_const_name in TIMEFRAMES:
            timeframe_const = self.TIMEFRAME_MAP[timeframe_const_name]

            try:
                if incremental:
                    fetched, inserted, invalid = self.fetch_incremental_data(
                        timeframe_name, timeframe_const
                    )
                else:
                    fetched, inserted, invalid = self.fetch_historical_data(
                        timeframe_name, timeframe_const
                    )

                results[timeframe_name] = {
                    'fetched':  fetched,
                    'inserted': inserted,
                    'invalid':  invalid
                }

            except Exception as e:
                self.collection_logger.log_error(
                    f"Error collecting {timeframe_name}", e
                )
                results[timeframe_name] = {
                    'fetched':  0,
                    'inserted': 0,
                    'invalid':  0,
                    'error':    str(e)
                }

        return results

    def run(
        self,
        setup_db: bool = True,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Main entry point: Initialize MT5, setup database, and collect data.
        """
        results = {
            'success':    False,
            'message':    '',
            'timeframes': {},
            'stats':      {}
        }

        mode = 'incremental' if incremental else 'full'
        self.collection_logger.start_collection(self.symbol, mode)

        try:
            if setup_db:
                logger.info("Setting up database schema...")
                self.db.setup_schema()

            if not self.initialize():
                results['message'] = "Failed to initialize MT5"
                self.collection_logger.log_error(results['message'])
                return results

            if not self.check_symbol():
                results['message'] = f"Symbol '{self.symbol}' not available"
                self.collection_logger.log_error(results['message'])
                return results

            logger.info("Starting data collection...")
            results['timeframes'] = self.collect_all_timeframes(incremental=incremental)

            results['summary'] = self.db.get_summary()
            results['session_summary'] = self.db.get_session_summary()

            results['success'] = True
            results['message'] = "Data collection completed successfully"

        except Exception as e:
            results['message'] = f"Error during data collection: {e}"
            self.collection_logger.log_error(results['message'], e)

        finally:
            self.shutdown()
            results['stats'] = self.collection_logger.end_collection(results['success'])

        return results


def main():
    """Main function to run the data collector"""
    import argparse

    parser = argparse.ArgumentParser(description='MT5 OHLCV Data Collector')
    parser.add_argument(
        '--full',
        action='store_true',
        help='Fetch full history instead of incremental updates'
    )
    parser.add_argument(
        '--skip-db-setup',
        action='store_true',
        help='Skip database schema setup'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Symbol to collect data for (default: {SYMBOL})'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable data validation (faster but less safe)'
    )
    parser.add_argument(
        '--broker-offset',
        type=int,
        default=BROKER_UTC_OFFSET,
        help=f'Broker UTC offset in hours (default: {BROKER_UTC_OFFSET}). '
             f'Use 2 for GMT+2, 3 for GMT+3'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce console output'
    )

    args = parser.parse_args()

    if args.quiet:
        import logging
        logging.getLogger('mt5_collector').setLevel(logging.WARNING)

    print("=" * 60)
    print("MT5 OHLCV Data Collector")
    print("=" * 60)
    print(f"Symbol:         {args.symbol}")
    print(f"Mode:           {'Full History' if args.full else 'Incremental'}")
    print(f"Database Setup: {'Skipped' if args.skip_db_setup else 'Enabled'}")
    print(f"Validation:     {'Disabled' if args.no_validation else 'Enabled'}")
    print(f"Broker Offset:  GMT+{args.broker_offset}")
    print(f"Sessions:       asian | london | london_ny_overlap | new_york | off_hours")
    print("=" * 60)

    collector = MT5Collector(
        symbol=args.symbol,
        enable_validation=not args.no_validation,
        broker_utc_offset=args.broker_offset
    )
    results = collector.run(
        setup_db=not args.skip_db_setup,
        incremental=not args.full
    )

    print("\n" + "=" * 60)
    print("Collection Results")
    print("=" * 60)

    if results['success']:
        print("Status: SUCCESS")
        print("\nTimeframe Summary:")
        print("-" * 40)

        total_fetched  = 0
        total_inserted = 0
        total_invalid  = 0

        for tf, data in results['timeframes'].items():
            fetched  = data.get('fetched', 0)
            inserted = data.get('inserted', 0)
            invalid  = data.get('invalid', 0)
            total_fetched  += fetched
            total_inserted += inserted
            total_invalid  += invalid

            status = f"Fetched {fetched:,}, Inserted {inserted:,}"
            if invalid > 0:
                status += f", Invalid {invalid:,}"
            print(f"  {tf:8s}: {status}")

        print("-" * 40)
        summary_line = f"  Total: Fetched {total_fetched:,}, Inserted {total_inserted:,}"
        if total_invalid > 0:
            summary_line += f", Invalid {total_invalid:,}"
        print(summary_line)

        # Session breakdown
        if 'session_summary' in results and results['session_summary']:
            print("\nSession Breakdown (all timeframes):")
            print("-" * 40)
            for row in results['session_summary']:
                print(f"  {row['session']:20s}: {row['total_candles']:>10,} candles  ({row['pct']}%)")

        # Database summary
        if 'summary' in results:
            print("\nDatabase Summary:")
            print("-" * 40)
            for row in results['summary']:
                print(f"  {row['timeframe']:8s}: {row['total_candles']:,} candles")
                print(f"           From: {row['earliest']}")
                print(f"           To:   {row['latest']}")
    else:
        print(f"Status: FAILED")
        print(f"Message: {results['message']}")

    if 'stats' in results and results['stats'].get('errors'):
        print("\nErrors encountered:")
        for err in results['stats']['errors'][:5]:
            print(f"  - {err['message']}")

    print("=" * 60)
    print("Log files: ./logs/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()