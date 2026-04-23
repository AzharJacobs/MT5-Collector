"""
Validators package
Exports the full DataValidator (backward-compatible) and individual checker classes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validators.schema_validator import ValidationResult, BatchValidationResult, SchemaValidator
from validators.anomaly_detector import AnomalyDetector
from validators.duplicate_checker import DuplicateChecker
from validators.gap_checker import GapChecker, EXPECTED_CANDLES_PER_DAY

from typing import List, Dict, Any
from logger import get_logger

logger = get_logger('mt5_collector.validators')


class DataValidator(SchemaValidator):
    """
    Full OHLCV data validator.
    Combines schema, anomaly, and derived-field checks into one interface.
    Backward-compatible with the original validator.py DataValidator.
    """

    def __init__(
        self,
        max_price: float = 1_000_000,
        min_price: float = 0,
        max_volume: float = 1_000_000_000,
        max_candle_size_pct: float = 50.0,
        check_outliers: bool = True,
        outlier_std_threshold: float = 5.0,
    ):
        self._anomaly = AnomalyDetector(
            max_price=max_price,
            min_price=min_price,
            max_volume=max_volume,
            max_candle_size_pct=max_candle_size_pct,
            outlier_std_threshold=outlier_std_threshold,
        )
        self.check_outliers = check_outliers

    def validate_candle(self, candle: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True, candle_data=candle)

        self.check_required_fields(candle, result)
        if not result.is_valid:
            return result

        self.check_data_types(candle, result)
        if not result.is_valid:
            return result

        self._anomaly.check_ohlcv_logic(candle, result)
        self._anomaly.check_price_range(candle, result)
        self._anomaly.check_volume(candle, result)
        self._anomaly.check_timestamp(candle, result)
        self.check_derived_fields(candle, result)
        self.check_categorical_fields(candle, result)

        if self.check_outliers:
            self._anomaly.check_outliers(candle, result)

        return result

    def validate_batch(
        self,
        candles: List[Dict[str, Any]],
        stop_on_first_error: bool = False,
    ) -> BatchValidationResult:
        batch = BatchValidationResult(total_count=len(candles))

        for candle in candles:
            result = self.validate_candle(candle)

            if result.is_valid:
                batch.valid_count += 1
                batch.valid_candles.append(candle)
            else:
                batch.invalid_count += 1
                batch.invalid_candles.append((candle, result.errors))
                batch.all_errors.extend(result.errors)
                if stop_on_first_error:
                    break

            if result.warnings:
                batch.warning_count += 1
                batch.all_warnings.extend(result.warnings)

        if batch.invalid_count > 0:
            logger.warning(
                f"Batch validation: {batch.invalid_count}/{batch.total_count} invalid"
            )

        return batch


def validate_candles(candles: List[Dict[str, Any]], **kwargs) -> BatchValidationResult:
    """Convenience function — validate a batch using DataValidator."""
    return DataValidator(**kwargs).validate_batch(candles)
