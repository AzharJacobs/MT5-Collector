"""
Calendar Collector
Fetches trading days, market holidays, and earnings dates.
Placeholder — implement when calendar data source is chosen.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from typing import List, Set

from logger import get_logger

logger = get_logger('mt5_collector.collectors.calendar')

# US market holidays (NYSE) — extend annually
US_HOLIDAYS_2024 = {
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
}

US_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

ALL_HOLIDAYS: Set[date] = US_HOLIDAYS_2024 | US_HOLIDAYS_2025


class CalendarCollector:
    """Provides trading day lookups and market calendar utilities."""

    def __init__(self, holidays: Set[date] = None):
        self.holidays = holidays or ALL_HOLIDAYS

    def is_trading_day(self, d: date) -> bool:
        """Returns True if d is a weekday and not a market holiday."""
        return d.weekday() < 5 and d not in self.holidays

    def get_trading_days(self, start: date, end: date) -> List[date]:
        """Return all trading days in [start, end] inclusive."""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        logger.debug(f"Trading days {start} → {end}: {len(days)} days")
        return days

    def get_missing_days(self, collected_days: Set[date], start: date, end: date) -> List[date]:
        """Return trading days in range that are missing from collected_days."""
        expected = set(self.get_trading_days(start, end))
        missing  = sorted(expected - collected_days)
        if missing:
            logger.warning(f"Missing {len(missing)} trading days between {start} and {end}")
        return missing
