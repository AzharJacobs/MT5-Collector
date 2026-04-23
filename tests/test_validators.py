"""
Tests for the validators package.
Verifies that validation logic correctly catches bad data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime

from validators import DataValidator, validate_candles
from validators.duplicate_checker import DuplicateChecker
from validators.gap_checker import GapChecker, EXPECTED_CANDLES_PER_DAY


def _valid_candle(**overrides) -> dict:
    base = {
        "symbol":      "USTECm",
        "timeframe":   "1H",
        "timestamp":   datetime(2024, 3, 18, 14, 0),
        "date":        datetime(2024, 3, 18).date(),
        "time":        datetime(2024, 3, 18, 14, 0).time(),
        "hour":        14,
        "day_of_week": "Monday",
        "month":       3,
        "year":        2024,
        "open":        100.0,
        "high":        105.0,
        "low":         99.0,
        "close":       103.0,
        "volume":      10000,
        "direction":   "buy",
        "candle_size": 6.0,
        "body_size":   3.0,
        "wick_upper":  2.0,
        "wick_lower":  1.0,
    }
    base.update(overrides)
    return base


class TestDataValidatorValid(unittest.TestCase):

    def setUp(self):
        self.validator = DataValidator()

    def test_valid_candle_passes(self):
        result = self.validator.validate_candle(_valid_candle())
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])

    def test_valid_batch_passes(self):
        candles = [_valid_candle(), _valid_candle(hour=15, close=101.0, direction="sell",
                                                   body_size=1.0, wick_upper=4.0)]
        batch = self.validator.validate_batch(candles)
        self.assertEqual(batch.valid_count, 2)
        self.assertEqual(batch.invalid_count, 0)


class TestDataValidatorInvalid(unittest.TestCase):

    def setUp(self):
        self.validator = DataValidator()

    def test_high_less_than_low(self):
        candle = _valid_candle(high=98.0, low=102.0)
        result = self.validator.validate_candle(candle)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("High" in e and "Low" in e for e in result.errors))

    def test_missing_required_field(self):
        candle = _valid_candle()
        del candle["close"]
        result = self.validator.validate_candle(candle)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("close" in e for e in result.errors))

    def test_null_required_field(self):
        candle = _valid_candle(open=None)
        result = self.validator.validate_candle(candle)
        self.assertFalse(result.is_valid)

    def test_invalid_direction(self):
        candle = _valid_candle(direction="sideways")
        result = self.validator.validate_candle(candle)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("direction" in e.lower() for e in result.errors))

    def test_negative_volume(self):
        candle = _valid_candle(volume=-100)
        result = self.validator.validate_candle(candle)
        self.assertFalse(result.is_valid)

    def test_future_timestamp(self):
        from datetime import timedelta
        future = datetime.now() + timedelta(days=10)
        candle = _valid_candle(timestamp=future, year=future.year, month=future.month, hour=future.hour)
        result = self.validator.validate_candle(candle)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("timestamp" in e.lower() or "future" in e.lower() for e in result.errors))

    def test_batch_counts_invalid(self):
        bad = _valid_candle(high=90.0, low=102.0)
        good = _valid_candle()
        batch = self.validator.validate_batch([good, bad, good])
        self.assertEqual(batch.valid_count, 2)
        self.assertEqual(batch.invalid_count, 1)


class TestDuplicateChecker(unittest.TestCase):

    def test_no_duplicates(self):
        candles = [
            _valid_candle(timestamp=datetime(2024, 1, 1, 10, 0)),
            _valid_candle(timestamp=datetime(2024, 1, 1, 11, 0)),
        ]
        unique, dupes = DuplicateChecker.find_duplicates(candles)
        self.assertEqual(len(unique), 2)
        self.assertEqual(len(dupes), 0)

    def test_finds_duplicates(self):
        ts = datetime(2024, 1, 1, 10, 0)
        candles = [_valid_candle(timestamp=ts), _valid_candle(timestamp=ts)]
        unique, dupes = DuplicateChecker.find_duplicates(candles)
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(dupes), 1)

    def test_deduplicate_convenience(self):
        ts = datetime(2024, 1, 1, 10, 0)
        candles = [_valid_candle(timestamp=ts)] * 5
        result = DuplicateChecker.deduplicate(candles)
        self.assertEqual(len(result), 1)


class TestValidateCandles(unittest.TestCase):

    def test_convenience_function(self):
        candles = [_valid_candle(), _valid_candle(high=80.0, low=102.0)]
        batch = validate_candles(candles)
        self.assertEqual(batch.total_count, 2)
        self.assertEqual(batch.valid_count, 1)
        self.assertEqual(batch.invalid_count, 1)


if __name__ == "__main__":
    unittest.main()
