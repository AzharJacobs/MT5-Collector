"""
Anomaly Detector
Catches bad data: impossible OHLCV values, price spikes, zero values, outliers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

from validators.schema_validator import ValidationResult
from logger import get_logger

logger = get_logger('mt5_collector.validators.anomaly')


class AnomalyDetector:
    """Detects price/volume anomalies and statistical outliers."""

    def __init__(
        self,
        max_price: float = 1_000_000,
        min_price: float = 0,
        max_volume: float = 1_000_000_000,
        max_candle_size_pct: float = 50.0,
        outlier_std_threshold: float = 5.0,
    ):
        self.max_price = max_price
        self.min_price = min_price
        self.max_volume = max_volume
        self.max_candle_size_pct = max_candle_size_pct
        self.outlier_std_threshold = outlier_std_threshold
        self._price_history: List[float] = []
        self._volume_history: List[float] = []

    def check_ohlcv_logic(self, candle: Dict[str, Any], result: ValidationResult):
        try:
            open_p  = float(candle['open'])
            high_p  = float(candle['high'])
            low_p   = float(candle['low'])
            close_p = float(candle['close'])

            if high_p < low_p:
                result.add_error(f"High ({high_p}) < Low ({low_p}) — impossible OHLCV")
            if high_p < open_p:
                result.add_error(f"High ({high_p}) < Open ({open_p})")
            if high_p < close_p:
                result.add_error(f"High ({high_p}) < Close ({close_p})")
            if low_p > open_p:
                result.add_error(f"Low ({low_p}) > Open ({open_p})")
            if low_p > close_p:
                result.add_error(f"Low ({low_p}) > Close ({close_p})")
            if any(p <= 0 for p in [open_p, high_p, low_p, close_p]):
                result.add_error("Zero or negative price detected")

        except (KeyError, TypeError, ValueError) as e:
            result.add_error(f"Error checking OHLCV logic: {e}")

    def check_price_range(self, candle: Dict[str, Any], result: ValidationResult):
        try:
            prices = [float(candle[k]) for k in ('open', 'high', 'low', 'close')]
            for price in prices:
                if price < self.min_price:
                    result.add_error(f"Price {price} below minimum {self.min_price}")
                if price > self.max_price:
                    result.add_error(f"Price {price} above maximum {self.max_price}")

            if prices[1] > 0:
                candle_pct = ((prices[1] - prices[2]) / prices[2]) * 100
                if candle_pct > self.max_candle_size_pct:
                    result.add_warning(f"Large candle: {candle_pct:.2f}% range")

        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass

    def check_volume(self, candle: Dict[str, Any], result: ValidationResult):
        try:
            volume = float(candle['volume'])
            if volume < 0:
                result.add_error(f"Negative volume: {volume}")
            if volume > self.max_volume:
                result.add_warning(f"Unusually high volume: {volume}")
            if volume == 0:
                result.add_warning("Zero volume candle")
        except (KeyError, TypeError, ValueError):
            pass

    def check_timestamp(self, candle: Dict[str, Any], result: ValidationResult):
        try:
            timestamp = candle.get('timestamp')
            if isinstance(timestamp, datetime):
                now = datetime.now(tz=timezone.utc) if timestamp.tzinfo else datetime.now()
                if timestamp > now + timedelta(days=1):
                    result.add_error(f"Future timestamp: {timestamp}")
                if timestamp.year < 1990:
                    result.add_error(f"Timestamp too old: {timestamp}")

            hour = candle.get('hour')
            if hour is not None and (hour < 0 or hour > 23):
                result.add_error(f"Invalid hour: {hour}")

            month = candle.get('month')
            if month is not None and (month < 1 or month > 12):
                result.add_error(f"Invalid month: {month}")

            year = candle.get('year')
            if year is not None and (year < 1990 or year > 2100):
                result.add_error(f"Invalid year: {year}")

        except Exception as e:
            result.add_error(f"Timestamp validation error: {e}")

    def check_outliers(self, candle: Dict[str, Any], result: ValidationResult):
        try:
            close_p = float(candle['close'])
            volume  = float(candle['volume'])

            self._price_history.append(close_p)
            self._volume_history.append(volume)

            if len(self._price_history) < 100:
                return

            if len(self._price_history) > 1000:
                self._price_history  = self._price_history[-1000:]
                self._volume_history = self._volume_history[-1000:]

            import statistics
            price_mean = statistics.mean(self._price_history[:-1])
            price_std  = statistics.stdev(self._price_history[:-1])
            if price_std > 0:
                z = abs(close_p - price_mean) / price_std
                if z > self.outlier_std_threshold:
                    result.add_warning(f"Price outlier: {close_p} (z={z:.2f})")

            vol_mean = statistics.mean(self._volume_history[:-1])
            vol_std  = statistics.stdev(self._volume_history[:-1])
            if vol_std > 0:
                z = abs(volume - vol_mean) / vol_std
                if z > self.outlier_std_threshold:
                    result.add_warning(f"Volume outlier: {volume} (z={z:.2f})")

        except (Exception,):
            pass
