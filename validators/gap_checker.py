"""
Gap Checker
Finds missing trading days or bars in collected data and flags them for refetching.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, date
from typing import List, Tuple

from logger import get_logger

logger = get_logger('mt5_collector.validators.gaps')

# Expected candles per trading day per timeframe (market hours ~6.5hrs for USTECm)
EXPECTED_CANDLES_PER_DAY = {
    "5min":  78,
    "15min": 26,
    "1H":    7,
    "4H":    2,
}


class GapChecker:
    """
    Queries the database to find days with suspiciously low candle counts,
    then provides windows to refetch for each gap.
    """

    def __init__(self, db_cursor_fn, symbol: str):
        """
        Args:
            db_cursor_fn: A context-manager callable that yields a DB cursor.
                          Typically DatabaseManager.get_cursor from storage.db_writer.
            symbol: The trading symbol to check.
        """
        self.get_cursor = db_cursor_fn
        self.symbol = symbol

    def find_gaps(
        self,
        timeframe_name: str,
        threshold: float = 0.5,
    ) -> List[Tuple[date, int, int]]:
        """
        Returns list of (day, actual_count, expected_count) for days that have
        less than `threshold` * expected candles. Skips weekends automatically.
        """
        query = """
            SELECT timestamp::date AS day, COUNT(*) AS cnt
            FROM ustech_ohlcv
            WHERE timeframe = %s AND symbol = %s
              AND EXTRACT(DOW FROM timestamp) NOT IN (0, 6)
            GROUP BY day
            ORDER BY day
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (timeframe_name, self.symbol))
            rows = cursor.fetchall()

        expected = EXPECTED_CANDLES_PER_DAY.get(timeframe_name, 0)
        if not expected:
            return []

        gaps = [
            (day, int(count), expected)
            for day, count in rows
            if count < expected * threshold
        ]

        if gaps:
            logger.warning(f"{timeframe_name}: {len(gaps)} days with <{threshold*100:.0f}% expected candles")
            for day, cnt, exp in gaps[:10]:
                logger.warning(f"  {day}: {cnt}/{exp} candles")

        return gaps

    def get_refetch_windows(
        self,
        timeframe_name: str,
        padding_days: int = 1,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Returns (window_start, window_end) pairs for each gap day,
        adding padding_days on each side to ensure full coverage.
        """
        gaps = self.find_gaps(timeframe_name)
        if not gaps:
            logger.info(f"{timeframe_name}: no gaps found")
            return []

        windows = []
        for gap_day, _, _ in gaps:
            start = datetime.combine(gap_day, datetime.min.time()) - timedelta(days=padding_days)
            end   = datetime.combine(gap_day, datetime.min.time()) + timedelta(days=padding_days + 1)
            windows.append((start, end))
            logger.info(f"  Gap window: {gap_day} → refetch {start.date()} to {end.date()}")

        return windows
