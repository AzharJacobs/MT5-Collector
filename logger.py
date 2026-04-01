"""
Production Logging Module
Configures rotating file logs and console output for production use
"""

import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Optional

# Create logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)


class LoggerConfig:
    """Configure production-ready logging with rotation"""

    # Log format with detailed information
    DETAILED_FORMAT = (
        '%(asctime)s | %(levelname)-8s | %(name)-20s | '
        '%(filename)s:%(lineno)d | %(message)s'
    )

    # Simpler format for console
    CONSOLE_FORMAT = '%(asctime)s | %(levelname)-8s | %(message)s'

    # Date format
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(
        self,
        log_dir: str = LOGS_DIR,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 10,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.console_level = console_level
        self.file_level = file_level

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

    def get_file_handler(
        self,
        filename: str,
        level: int = None
    ) -> RotatingFileHandler:
        """Create a rotating file handler"""
        filepath = os.path.join(self.log_dir, filename)
        handler = RotatingFileHandler(
            filepath,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level or self.file_level)
        handler.setFormatter(logging.Formatter(
            self.DETAILED_FORMAT,
            datefmt=self.DATE_FORMAT
        ))
        return handler

    def get_timed_handler(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        level: int = None
    ) -> TimedRotatingFileHandler:
        """Create a time-based rotating file handler"""
        filepath = os.path.join(self.log_dir, filename)
        handler = TimedRotatingFileHandler(
            filepath,
            when=when,
            interval=interval,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level or self.file_level)
        handler.setFormatter(logging.Formatter(
            self.DETAILED_FORMAT,
            datefmt=self.DATE_FORMAT
        ))
        return handler

    def get_console_handler(self, level: int = None) -> logging.StreamHandler:
        """Create a console handler"""
        handler = logging.StreamHandler()
        handler.setLevel(level or self.console_level)
        handler.setFormatter(logging.Formatter(
            self.CONSOLE_FORMAT,
            datefmt=self.DATE_FORMAT
        ))
        return handler

    def get_error_handler(self) -> RotatingFileHandler:
        """Create a dedicated error log handler"""
        return self.get_file_handler('error.log', level=logging.ERROR)


def setup_logging(
    name: str = 'mt5_collector',
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_error_log: bool = True
) -> logging.Logger:
    """
    Setup production logging with rotation.

    Creates:
    - Console output (INFO level by default)
    - Main log file with size-based rotation (DEBUG level)
    - Daily log file with time-based rotation
    - Error log file (ERROR level only)

    Args:
        name: Logger name
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        enable_error_log: Whether to create separate error log

    Returns:
        Configured logger instance
    """
    config = LoggerConfig(
        console_level=console_level,
        file_level=file_level
    )

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add console handler
    logger.addHandler(config.get_console_handler())

    # Add main rotating file handler (size-based)
    logger.addHandler(config.get_file_handler('mt5_collector.log'))

    # Add daily rotating handler (time-based)
    logger.addHandler(config.get_timed_handler(
        'mt5_daily.log',
        when='midnight',
        interval=1
    ))

    # Add error-only handler
    if enable_error_log:
        logger.addHandler(config.get_error_handler())

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. If name is None, returns the root mt5_collector logger.

    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
    """
    if name is None:
        name = 'mt5_collector'

    # Check if root logger is already configured
    root_logger = logging.getLogger('mt5_collector')
    if not root_logger.handlers:
        setup_logging()

    return logging.getLogger(name)


class CollectionLogger:
    """
    Specialized logger for data collection operations.
    Tracks statistics and creates structured log entries.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger('mt5_collector.collection')
        self.stats = {
            'start_time': None,
            'end_time': None,
            'timeframes_processed': 0,
            'total_fetched': 0,
            'total_inserted': 0,
            'total_duplicates': 0,
            'total_invalid': 0,
            'errors': []
        }

    def start_collection(self, symbol: str, mode: str = 'incremental'):
        """Log start of collection session"""
        self.stats['start_time'] = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info(f"COLLECTION STARTED")
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Time: {self.stats['start_time']}")
        self.logger.info("=" * 60)

    def log_timeframe_start(self, timeframe: str):
        """Log start of timeframe processing"""
        self.logger.info(f"Processing timeframe: {timeframe}")

    def log_chunk_processed(
        self,
        timeframe: str,
        fetched: int,
        inserted: int,
        invalid: int = 0,
        earliest: str = None
    ):
        """Log chunk processing results"""
        self.stats['total_fetched'] += fetched
        self.stats['total_inserted'] += inserted
        self.stats['total_duplicates'] += (fetched - inserted - invalid)
        self.stats['total_invalid'] += invalid

        self.logger.debug(
            f"  {timeframe}: Chunk processed - "
            f"Fetched: {fetched}, Inserted: {inserted}, Invalid: {invalid}"
            f"{f', Earliest: {earliest}' if earliest else ''}"
        )

    def log_timeframe_complete(
        self,
        timeframe: str,
        fetched: int,
        inserted: int
    ):
        """Log completion of timeframe"""
        self.stats['timeframes_processed'] += 1
        self.logger.info(
            f"Completed {timeframe}: "
            f"Fetched {fetched:,}, Inserted {inserted:,} new records"
        )

    def log_error(self, message: str, exception: Exception = None):
        """Log an error"""
        self.stats['errors'].append({
            'time': datetime.now(),
            'message': message,
            'exception': str(exception) if exception else None
        })
        if exception:
            self.logger.error(f"{message}: {exception}", exc_info=True)
        else:
            self.logger.error(message)

    def log_warning(self, message: str):
        """Log a warning"""
        self.logger.warning(message)

    def end_collection(self, success: bool = True):
        """Log end of collection session with summary"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']

        self.logger.info("=" * 60)
        self.logger.info(f"COLLECTION {'COMPLETED' if success else 'FAILED'}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Timeframes processed: {self.stats['timeframes_processed']}")
        self.logger.info(f"Total fetched: {self.stats['total_fetched']:,}")
        self.logger.info(f"Total inserted: {self.stats['total_inserted']:,}")
        self.logger.info(f"Total duplicates skipped: {self.stats['total_duplicates']:,}")
        self.logger.info(f"Total invalid records: {self.stats['total_invalid']:,}")

        if self.stats['errors']:
            self.logger.info(f"Errors encountered: {len(self.stats['errors'])}")
            for err in self.stats['errors']:
                self.logger.info(f"  - {err['message']}")

        self.logger.info("=" * 60)

        return self.stats


# Initialize logging when module is imported
_default_logger = None

def init_default_logger():
    """Initialize the default logger (called automatically)"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


if __name__ == "__main__":
    # Test logging configuration
    logger = setup_logging()

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print(f"\nLog files created in: {LOGS_DIR}")
    print("Check the following files:")
    print("  - mt5_collector.log (main log with rotation)")
    print("  - mt5_daily.log (daily rotation)")
    print("  - error.log (errors only)")
