"""
Schema Validator
Confirms correct columns, data types, and categorical field values exist.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from decimal import InvalidOperation

from logger import get_logger

logger = get_logger('mt5_collector.validators.schema')


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


class SchemaValidator:
    """Validates that required fields, data types, and categorical values are correct."""

    REQUIRED_FIELDS = [
        'symbol', 'timeframe', 'timestamp', 'date', 'time',
        'hour', 'day_of_week', 'month', 'year',
        'open', 'high', 'low', 'close', 'volume',
        'direction', 'candle_size', 'body_size',
        'wick_upper', 'wick_lower'
    ]

    VALID_DIRECTIONS = ['buy', 'sell', 'neutral']
    VALID_DAYS = [
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]
    VALID_TIMEFRAMES = ['5min', '15min', '1H', '4H']

    def check_required_fields(self, candle: Dict[str, Any], result: ValidationResult):
        for f in self.REQUIRED_FIELDS:
            if f not in candle:
                result.add_error(f"Missing required field: {f}")
            elif candle[f] is None:
                result.add_error(f"Null value for required field: {f}")

    def check_data_types(self, candle: Dict[str, Any], result: ValidationResult):
        numeric_fields = [
            'open', 'high', 'low', 'close', 'volume',
            'candle_size', 'body_size', 'wick_upper', 'wick_lower'
        ]
        for f in numeric_fields:
            value = candle.get(f)
            if value is not None:
                try:
                    float(value)
                except (ValueError, TypeError, InvalidOperation):
                    result.add_error(f"Invalid numeric value for {f}: {value}")

        int_fields = ['hour', 'month', 'year']
        for f in int_fields:
            value = candle.get(f)
            if value is not None and not isinstance(value, int):
                try:
                    int(value)
                except (ValueError, TypeError):
                    result.add_error(f"Invalid integer value for {f}: {value}")

        str_fields = ['symbol', 'timeframe', 'direction', 'day_of_week']
        for f in str_fields:
            value = candle.get(f)
            if value is not None and not isinstance(value, str):
                result.add_error(f"Invalid string value for {f}: {value}")

    def check_categorical_fields(self, candle: Dict[str, Any], result: ValidationResult):
        direction = candle.get('direction')
        if direction and direction not in self.VALID_DIRECTIONS:
            result.add_error(f"Invalid direction: {direction}")

        day = candle.get('day_of_week')
        if day and day not in self.VALID_DAYS:
            result.add_error(f"Invalid day_of_week: {day}")

        timeframe = candle.get('timeframe')
        if timeframe and timeframe not in self.VALID_TIMEFRAMES:
            result.add_warning(f"Unknown timeframe: {timeframe}")

    def check_derived_fields(self, candle: Dict[str, Any], result: ValidationResult):
        try:
            open_p  = float(candle['open'])
            high_p  = float(candle['high'])
            low_p   = float(candle['low'])
            close_p = float(candle['close'])

            candle_size = float(candle['candle_size'])
            body_size   = float(candle['body_size'])
            wick_upper  = float(candle['wick_upper'])
            wick_lower  = float(candle['wick_lower'])

            if abs(candle_size - (high_p - low_p)) > 0.0001:
                result.add_warning(
                    f"Candle size mismatch: got {candle_size}, expected {high_p - low_p}"
                )
            if abs(body_size - abs(close_p - open_p)) > 0.0001:
                result.add_warning(
                    f"Body size mismatch: got {body_size}, expected {abs(close_p - open_p)}"
                )
            if wick_upper < -0.0001:
                result.add_error(f"Negative upper wick: {wick_upper}")
            if wick_lower < -0.0001:
                result.add_error(f"Negative lower wick: {wick_lower}")

            direction = candle.get('direction')
            if close_p > open_p and direction != 'buy':
                result.add_warning(f"Direction mismatch: close > open but direction='{direction}'")
            elif close_p < open_p and direction != 'sell':
                result.add_warning(f"Direction mismatch: close < open but direction='{direction}'")
            elif close_p == open_p and direction != 'neutral':
                result.add_warning(f"Direction mismatch: close == open but direction='{direction}'")

        except (KeyError, TypeError, ValueError):
            pass
