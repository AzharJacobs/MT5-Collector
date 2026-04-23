"""
Tests for the storage package.
Verifies that data is written and read correctly.
Uses mocked DB connections — does not require a live PostgreSQL instance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from storage.db_writer import DatabaseManager
from storage.db_reader import DataReader
from validators.duplicate_checker import DuplicateChecker


def _make_candle(**overrides) -> dict:
    base = {
        "symbol": "USTECm", "timeframe": "1H",
        "timestamp": datetime(2024, 3, 18, 14, 0),
        "date": datetime(2024, 3, 18).date(),
        "time": datetime(2024, 3, 18, 14, 0).time(),
        "hour": 14, "day_of_week": "Monday", "month": 3, "year": 2024,
        "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0,
        "volume": 10000, "spread": 2, "direction": "buy",
        "candle_size": 6.0, "body_size": 3.0,
        "wick_upper": 2.0, "wick_lower": 1.0, "session": "london",
    }
    base.update(overrides)
    return base


class TestDatabaseManagerInsert(unittest.TestCase):

    @patch('storage.db_writer.psycopg2.connect')
    def test_insert_empty_returns_zero(self, mock_connect):
        db = DatabaseManager()
        result = db.insert_candles([])
        self.assertEqual(result, 0)
        mock_connect.assert_not_called()

    @patch('storage.db_writer.execute_values')
    @patch('storage.db_writer.psycopg2.connect')
    def test_insert_calls_execute_values(self, mock_connect, mock_execute):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = DatabaseManager()
        candles = [_make_candle()]
        db.insert_candles(candles)

        mock_execute.assert_called_once()


class TestDatabaseManagerRead(unittest.TestCase):

    @patch('storage.db_writer.psycopg2.connect')
    def test_get_latest_timestamp_returns_none_on_empty(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        db = DatabaseManager()
        with patch.object(db, 'get_cursor') as mock_get_cursor:
            mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)
            result = db.get_latest_timestamp("USTECm", "1H")

        self.assertIsNone(result)

    @patch('storage.db_writer.psycopg2.connect')
    def test_get_row_count(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        db = DatabaseManager()
        with patch.object(db, 'get_cursor') as mock_gc:
            mock_gc.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)
            count = db.get_row_count()

        self.assertEqual(count, 42)


class TestDuplicateCheckerIntegration(unittest.TestCase):
    """Verifies that duplicates are removed before insertion (storage integration)."""

    def test_dedup_before_insert(self):
        ts = datetime(2024, 1, 1, 10, 0)
        candles = [_make_candle(timestamp=ts)] * 3
        unique = DuplicateChecker.deduplicate(candles)
        self.assertEqual(len(unique), 1)


if __name__ == "__main__":
    unittest.main()
