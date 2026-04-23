"""
Duplicate Checker
Finds and removes duplicate timestamps from a batch of candles before DB insertion.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple

from logger import get_logger

logger = get_logger('mt5_collector.validators.duplicates')


class DuplicateChecker:
    """Detects and removes duplicate (symbol, timeframe, timestamp) entries in a batch."""

    @staticmethod
    def find_duplicates(
        candles: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits candles into unique and duplicate lists.
        Keeps the first occurrence of each (symbol, timeframe, timestamp) key.
        Returns (unique_candles, duplicate_candles).
        """
        seen = set()
        unique = []
        duplicates = []

        for candle in candles:
            key = (
                candle.get('symbol'),
                candle.get('timeframe'),
                candle.get('timestamp'),
            )
            if key in seen:
                duplicates.append(candle)
            else:
                seen.add(key)
                unique.append(candle)

        if duplicates:
            logger.warning(
                f"DuplicateChecker: {len(duplicates)} duplicates found in batch of {len(candles)}"
            )

        return unique, duplicates

    @staticmethod
    def deduplicate(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convenience: return only unique candles, discarding duplicates."""
        unique, _ = DuplicateChecker.find_duplicates(candles)
        return unique
