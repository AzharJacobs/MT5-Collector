"""
Tests for the collectors package.
Verifies that data fetching logic works correctly (unit-level, no live MT5 needed).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, date
import pandas as pd

from collectors.price_collector import MT5Collector, get_session, parse_date
from collectors.calendar_collector import CalendarCollector


class TestGetSession(unittest.TestCase):

    def test_asian_session(self):
        # UTC 02:00 = broker 05:00 (offset +3) → asian
        ts = pd.Timestamp("2024-01-15 05:00:00")
        self.assertEqual(get_session(ts, broker_utc_offset=3), "asian")

    def test_london_session(self):
        ts = pd.Timestamp("2024-01-15 10:00:00")  # UTC 07:00
        self.assertEqual(get_session(ts, broker_utc_offset=3), "london")

    def test_ny_session(self):
        ts = pd.Timestamp("2024-01-15 19:00:00")  # UTC 16:00
        self.assertEqual(get_session(ts, broker_utc_offset=3), "new_york")

    def test_off_hours(self):
        ts = pd.Timestamp("2024-01-15 00:00:00")  # UTC 21:00 previous day
        self.assertEqual(get_session(ts, broker_utc_offset=3), "off_hours")


class TestParseDate(unittest.TestCase):

    def test_parses_iso_string(self):
        d = parse_date("2024-01-01", datetime(2020, 1, 1))
        self.assertEqual(d.year, 2024)
        self.assertEqual(d.month, 1)
        self.assertEqual(d.day, 1)

    def test_returns_datetime_unchanged(self):
        dt = datetime(2024, 6, 15)
        self.assertEqual(parse_date(dt, datetime(2020, 1, 1)), dt)

    def test_returns_default_on_bad_input(self):
        default = datetime(2020, 1, 1)
        result = parse_date("not-a-date", default)
        self.assertEqual(result, default)


class TestCandleBuilder(unittest.TestCase):
    """Test MT5Collector._build_candles using a mock MT5 setup."""

    def _make_collector(self):
        with patch('collectors.price_collector.mt5') as mock_mt5:
            mock_mt5.TIMEFRAME_M5  = 5
            mock_mt5.TIMEFRAME_M15 = 15
            mock_mt5.TIMEFRAME_H1  = 16385
            mock_mt5.TIMEFRAME_H4  = 16388
            collector = MT5Collector.__new__(MT5Collector)
            collector.symbol             = "USTECm"
            collector.broker_utc_offset  = 3
            collector.enable_validation  = False
            collector.validator          = None
            collector.db                 = MagicMock()
            collector.collection_logger  = MagicMock()
            collector.data_start_date    = datetime(2024, 1, 1)
            collector.data_end_date      = datetime(2025, 1, 1, 23, 59, 59)
            collector._gap_checker       = MagicMock()
            return collector

    def test_build_candles_returns_correct_fields(self):
        collector = self._make_collector()

        # Create a minimal DataFrame row (Unix timestamp for 2024-01-02 10:00 UTC)
        ts_unix = int(datetime(2024, 1, 2, 10, 0).timestamp())
        df = pd.DataFrame([{
            "time": ts_unix, "open": 100.0, "high": 105.0,
            "low": 99.0, "close": 103.0, "tick_volume": 5000, "spread": 2
        }])

        candles = collector._build_candles(df, "1H")
        self.assertEqual(len(candles), 1)
        c = candles[0]

        self.assertEqual(c["symbol"], "USTECm")
        self.assertEqual(c["timeframe"], "1H")
        self.assertEqual(c["direction"], "buy")   # close > open
        self.assertAlmostEqual(c["candle_size"], 6.0)
        self.assertAlmostEqual(c["body_size"], 3.0)
        self.assertEqual(c["spread"], 2)

    def test_build_candles_sell_direction(self):
        collector = self._make_collector()
        ts_unix = int(datetime(2024, 1, 2, 14, 0).timestamp())
        df = pd.DataFrame([{
            "time": ts_unix, "open": 105.0, "high": 106.0,
            "low": 100.0, "close": 101.0, "tick_volume": 3000, "spread": 1
        }])
        candles = collector._build_candles(df, "1H")
        self.assertEqual(candles[0]["direction"], "sell")


class TestCalendarCollector(unittest.TestCase):

    def setUp(self):
        self.calendar = CalendarCollector()

    def test_weekends_not_trading_days(self):
        saturday = date(2024, 1, 6)
        sunday   = date(2024, 1, 7)
        self.assertFalse(self.calendar.is_trading_day(saturday))
        self.assertFalse(self.calendar.is_trading_day(sunday))

    def test_weekday_is_trading_day(self):
        monday = date(2024, 1, 8)
        self.assertTrue(self.calendar.is_trading_day(monday))

    def test_holiday_not_trading_day(self):
        new_years = date(2024, 1, 1)
        self.assertFalse(self.calendar.is_trading_day(new_years))

    def test_trading_days_count(self):
        days = self.calendar.get_trading_days(date(2024, 1, 1), date(2024, 1, 7))
        # Jan 1 = holiday, Jan 6/7 = weekend → only Jan 2,3,4,5
        self.assertEqual(len(days), 4)


if __name__ == "__main__":
    unittest.main()
