"""
Data Validation Module
Performs quality checks on OHLCV data before database insertion
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import logging

from logger import get_logger

logger = get_logger('mt5_collector.validator')


@dataclass
class ValidationResult:
    """Result of validation for a single candle"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    candle_data: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)


@dataclass
class BatchValidationResult:
    """Result of validation for a batch of candles"""
    total_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    warning_count: int = 0
    valid_candles: List[Dict[str, Any]] = field(default_factory=list)
    invalid_candles: List[Tuple[Dict[str, Any], List[str]]] = field(default_factory=list)
    all_errors: List[str] = field(default_factory=list)
    all_warnings: List[str] = field(default_factory=list)


class DataValidator:
    """
    Validates OHLCV candle data before database insertion.

    Checks performed:
    - Required fields presence
    - Data type validation
    - OHLCV logical consistency (high >= low, etc.)
    - Price range validation (no negative prices)
    - Timestamp validation
    - Volume validation
    - Outlier detection
    """

    # Required fields for each candle
    REQUIRED_FIELDS = [
        'symbol', 'timeframe', 'timestamp', 'date', 'time',
        'hour', 'day_of_week', 'month', 'year',
        'open', 'high', 'low', 'close', 'volume',
        'direction', 'candle_size', 'body_size',
        'wick_upper', 'wick_lower'
    ]

    # Valid directions
    VALID_DIRECTIONS = ['buy', 'sell', 'neutral']

    # Valid days of week
    VALID_DAYS = [
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]

    # Valid timeframes
    VALID_TIMEFRAMES = [
        '1min', '2min', '3min', '4min', '5min',
        '10min', '15min', '30min', '1H', '4H', '1D'
    ]

    def __init__(
        self,
        max_price: float = 1_000_000,
        min_price: float = 0,
        max_volume: float = 1_000_000_000,
        max_candle_size_pct: float = 50.0,  # Max % move in single candle
        check_outliers: bool = True,
        outlier_std_threshold: float = 5.0
    ):
        self.max_price = max_price
        self.min_price = min_price
        self.max_volume = max_volume
        self.max_candle_size_pct = max_candle_size_pct
        self.check_outliers = check_outliers
        self.outlier_std_threshold = outlier_std_threshold

        # Statistics for outlier detection
        self._price_history: List[float] = []
        self._volume_history: List[float] = []

    def validate_candle(self, candle: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single candle record.

        Args:
            candle: Dictionary containing candle data

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        result = ValidationResult(is_valid=True, candle_data=candle)

        # Check required fields
        self._check_required_fields(candle, result)
        if not result.is_valid:
            return result

        # Validate data types
        self._check_data_types(candle, result)
        if not result.is_valid:
            return result

        # Validate OHLCV logic
        self._check_ohlcv_logic(candle, result)

        # Validate price ranges
        self._check_price_range(candle, result)

        # Validate volume
        self._check_volume(candle, result)

        # Validate timestamp
        self._check_timestamp(candle, result)

        # Validate derived fields
        self._check_derived_fields(candle, result)

        # Validate categorical fields
        self._check_categorical_fields(candle, result)

        # Check for potential outliers (warning only)
        if self.check_outliers:
            self._check_outliers(candle, result)

        return result

    def validate_batch(
        self,
        candles: List[Dict[str, Any]],
        stop_on_first_error: bool = False
    ) -> BatchValidationResult:
        """
        Validate a batch of candles.

        Args:
            candles: List of candle dictionaries
            stop_on_first_error: If True, stops validation on first invalid candle

        Returns:
            BatchValidationResult with aggregated results
        """
        batch_result = BatchValidationResult(total_count=len(candles))

        for candle in candles:
            result = self.validate_candle(candle)

            if result.is_valid:
                batch_result.valid_count += 1
                batch_result.valid_candles.append(candle)
            else:
                batch_result.invalid_count += 1
                batch_result.invalid_candles.append((candle, result.errors))
                batch_result.all_errors.extend(result.errors)

                if stop_on_first_error:
                    break

            if result.warnings:
                batch_result.warning_count += 1
                batch_result.all_warnings.extend(result.warnings)

        # Log summary
        if batch_result.invalid_count > 0:
            logger.warning(
                f"Batch validation: {batch_result.invalid_count}/{batch_result.total_count} "
                f"invalid records"
            )

        return batch_result

    def _check_required_fields(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Check all required fields are present"""
        for field in self.REQUIRED_FIELDS:
            if field not in candle:
                result.add_error(f"Missing required field: {field}")
            elif candle[field] is None:
                result.add_error(f"Null value for required field: {field}")

    def _check_data_types(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate data types for each field"""
        # Numeric fields
        numeric_fields = [
            'open', 'high', 'low', 'close', 'volume',
            'candle_size', 'body_size', 'wick_upper', 'wick_lower'
        ]

        for field in numeric_fields:
            value = candle.get(field)
            if value is not None:
                try:
                    float(value)
                except (ValueError, TypeError, InvalidOperation):
                    result.add_error(f"Invalid numeric value for {field}: {value}")

        # Integer fields
        int_fields = ['hour', 'month', 'year']
        for field in int_fields:
            value = candle.get(field)
            if value is not None:
                if not isinstance(value, int):
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        result.add_error(f"Invalid integer value for {field}: {value}")

        # String fields
        str_fields = ['symbol', 'timeframe', 'direction', 'day_of_week']
        for field in str_fields:
            value = candle.get(field)
            if value is not None and not isinstance(value, str):
                result.add_error(f"Invalid string value for {field}: {value}")

    def _check_ohlcv_logic(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Check OHLCV logical consistency"""
        try:
            open_p = float(candle['open'])
            high_p = float(candle['high'])
            low_p = float(candle['low'])
            close_p = float(candle['close'])

            # High must be >= Low
            if high_p < low_p:
                result.add_error(
                    f"High ({high_p}) < Low ({low_p}) - impossible OHLCV values"
                )

            # High must be >= Open and Close
            if high_p < open_p:
                result.add_error(f"High ({high_p}) < Open ({open_p})")
            if high_p < close_p:
                result.add_error(f"High ({high_p}) < Close ({close_p})")

            # Low must be <= Open and Close
            if low_p > open_p:
                result.add_error(f"Low ({low_p}) > Open ({open_p})")
            if low_p > close_p:
                result.add_error(f"Low ({low_p}) > Close ({close_p})")

            # Check for zero or negative prices
            if any(p <= 0 for p in [open_p, high_p, low_p, close_p]):
                result.add_error("Zero or negative price detected")

        except (KeyError, TypeError, ValueError) as e:
            result.add_error(f"Error checking OHLCV logic: {e}")

    def _check_price_range(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Check prices are within acceptable range"""
        try:
            prices = [
                float(candle['open']),
                float(candle['high']),
                float(candle['low']),
                float(candle['close'])
            ]

            for price in prices:
                if price < self.min_price:
                    result.add_error(f"Price {price} below minimum {self.min_price}")
                if price > self.max_price:
                    result.add_error(f"Price {price} above maximum {self.max_price}")

            # Check candle size percentage
            if prices[1] > 0:  # high > 0
                candle_pct = ((prices[1] - prices[2]) / prices[2]) * 100
                if candle_pct > self.max_candle_size_pct:
                    result.add_warning(
                        f"Large candle detected: {candle_pct:.2f}% range"
                    )

        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass  # Already caught in other checks

    def _check_volume(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate volume data"""
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

    def _check_timestamp(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate timestamp fields"""
        try:
            timestamp = candle.get('timestamp')

            # Check if timestamp is in the future
            if isinstance(timestamp, datetime):
                if timestamp > datetime.now() + timedelta(days=1):
                    result.add_error(f"Future timestamp detected: {timestamp}")

                # Check if timestamp is too old (before 1990)
                if timestamp.year < 1990:
                    result.add_error(f"Timestamp too old: {timestamp}")

            # Validate hour range
            hour = candle.get('hour')
            if hour is not None and (hour < 0 or hour > 23):
                result.add_error(f"Invalid hour value: {hour}")

            # Validate month range
            month = candle.get('month')
            if month is not None and (month < 1 or month > 12):
                result.add_error(f"Invalid month value: {month}")

            # Validate year range
            year = candle.get('year')
            if year is not None and (year < 1990 or year > 2100):
                result.add_error(f"Invalid year value: {year}")

        except Exception as e:
            result.add_error(f"Timestamp validation error: {e}")

    def _check_derived_fields(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate derived/calculated fields"""
        try:
            open_p = float(candle['open'])
            high_p = float(candle['high'])
            low_p = float(candle['low'])
            close_p = float(candle['close'])

            candle_size = float(candle['candle_size'])
            body_size = float(candle['body_size'])
            wick_upper = float(candle['wick_upper'])
            wick_lower = float(candle['wick_lower'])

            # Candle size should equal high - low
            expected_candle_size = high_p - low_p
            if abs(candle_size - expected_candle_size) > 0.0001:
                result.add_warning(
                    f"Candle size mismatch: got {candle_size}, "
                    f"expected {expected_candle_size}"
                )

            # Body size should equal abs(close - open)
            expected_body_size = abs(close_p - open_p)
            if abs(body_size - expected_body_size) > 0.0001:
                result.add_warning(
                    f"Body size mismatch: got {body_size}, "
                    f"expected {expected_body_size}"
                )

            # Wick values should not be negative
            if wick_upper < -0.0001:
                result.add_error(f"Negative upper wick: {wick_upper}")
            if wick_lower < -0.0001:
                result.add_error(f"Negative lower wick: {wick_lower}")

            # Direction consistency
            direction = candle.get('direction')
            if close_p > open_p and direction != 'buy':
                result.add_warning(
                    f"Direction mismatch: close > open but direction is '{direction}'"
                )
            elif close_p < open_p and direction != 'sell':
                result.add_warning(
                    f"Direction mismatch: close < open but direction is '{direction}'"
                )
            elif close_p == open_p and direction != 'neutral':
                result.add_warning(
                    f"Direction mismatch: close == open but direction is '{direction}'"
                )

        except (KeyError, TypeError, ValueError):
            pass

    def _check_categorical_fields(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate categorical field values"""
        # Direction
        direction = candle.get('direction')
        if direction and direction not in self.VALID_DIRECTIONS:
            result.add_error(f"Invalid direction: {direction}")

        # Day of week
        day = candle.get('day_of_week')
        if day and day not in self.VALID_DAYS:
            result.add_error(f"Invalid day_of_week: {day}")

        # Timeframe
        timeframe = candle.get('timeframe')
        if timeframe and timeframe not in self.VALID_TIMEFRAMES:
            result.add_warning(f"Unknown timeframe: {timeframe}")

    def _check_outliers(
        self,
        candle: Dict[str, Any],
        result: ValidationResult
    ):
        """Check for statistical outliers (warning only)"""
        try:
            close_p = float(candle['close'])
            volume = float(candle['volume'])

            # Add to history for future calculations
            self._price_history.append(close_p)
            self._volume_history.append(volume)

            # Only check if we have enough history
            if len(self._price_history) < 100:
                return

            # Keep only recent history
            if len(self._price_history) > 1000:
                self._price_history = self._price_history[-1000:]
                self._volume_history = self._volume_history[-1000:]

            # Calculate mean and std for prices
            import statistics
            price_mean = statistics.mean(self._price_history[:-1])
            price_std = statistics.stdev(self._price_history[:-1])

            if price_std > 0:
                price_z = abs(close_p - price_mean) / price_std
                if price_z > self.outlier_std_threshold:
                    result.add_warning(
                        f"Price outlier detected: {close_p} "
                        f"(z-score: {price_z:.2f})"
                    )

            # Calculate mean and std for volume
            vol_mean = statistics.mean(self._volume_history[:-1])
            vol_std = statistics.stdev(self._volume_history[:-1])

            if vol_std > 0:
                vol_z = abs(volume - vol_mean) / vol_std
                if vol_z > self.outlier_std_threshold:
                    result.add_warning(
                        f"Volume outlier detected: {volume} "
                        f"(z-score: {vol_z:.2f})"
                    )

        except (statistics.StatisticsError, ValueError, ZeroDivisionError):
            pass  # Not enough data for statistics


def validate_candles(
    candles: List[Dict[str, Any]],
    **validator_kwargs
) -> BatchValidationResult:
    """
    Convenience function to validate a batch of candles.

    Args:
        candles: List of candle dictionaries
        **validator_kwargs: Arguments to pass to DataValidator

    Returns:
        BatchValidationResult
    """
    validator = DataValidator(**validator_kwargs)
    return validator.validate_batch(candles)


if __name__ == "__main__":
    # Test validation with sample data
    test_candles = [
        # Valid candle
        {
            'symbol': 'USTech',
            'timeframe': '1H',
            'timestamp': datetime.now(),
            'date': datetime.now().date(),
            'time': datetime.now().time(),
            'hour': 14,
            'day_of_week': 'Monday',
            'month': 3,
            'year': 2024,
            'open': 100.0,
            'high': 105.0,
            'low': 99.0,
            'close': 103.0,
            'volume': 10000,
            'direction': 'buy',
            'candle_size': 6.0,
            'body_size': 3.0,
            'wick_upper': 2.0,
            'wick_lower': 1.0
        },
        # Invalid candle (high < low)
        {
            'symbol': 'USTech',
            'timeframe': '1H',
            'timestamp': datetime.now(),
            'date': datetime.now().date(),
            'time': datetime.now().time(),
            'hour': 15,
            'day_of_week': 'Monday',
            'month': 3,
            'year': 2024,
            'open': 100.0,
            'high': 98.0,  # Invalid: high < low
            'low': 102.0,
            'close': 101.0,
            'volume': 10000,
            'direction': 'buy',
            'candle_size': -4.0,
            'body_size': 1.0,
            'wick_upper': 0.0,
            'wick_lower': 0.0
        }
    ]

    result = validate_candles(test_candles)

    print(f"Total: {result.total_count}")
    print(f"Valid: {result.valid_count}")
    print(f"Invalid: {result.invalid_count}")
    print(f"Warnings: {result.warning_count}")

    if result.invalid_candles:
        print("\nInvalid candles:")
        for candle, errors in result.invalid_candles:
            print(f"  Timestamp: {candle.get('timestamp')}")
            for error in errors:
                print(f"    - {error}")
