"""
Fundamental Collector
Fetches PE ratio, EPS, revenue, and other fundamental data for tracked symbols.
Placeholder — implement when fundamental data source is chosen (e.g. yfinance, Alpha Vantage).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional
from logger import get_logger

logger = get_logger('mt5_collector.collectors.fundamental')


class FundamentalCollector:
    """
    Fetches fundamental data (PE, EPS, revenue, etc.) for a given symbol.

    To implement: choose a data provider and fill in fetch_fundamentals().
    Suggested providers: yfinance (free), Alpha Vantage (API key), Polygon.io.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch_fundamentals(self) -> Optional[Dict[str, Any]]:
        """
        Fetch latest fundamental data for self.symbol.
        Returns a dict with keys: pe_ratio, eps, revenue, market_cap, etc.
        Returns None if data is unavailable.
        """
        logger.warning(
            f"FundamentalCollector.fetch_fundamentals() not yet implemented for {self.symbol}. "
            "Choose a data provider and implement this method."
        )
        return None

    def fetch_earnings_dates(self) -> list:
        """
        Fetch upcoming and historical earnings dates.
        Returns a list of dicts: [{date, eps_estimate, eps_actual, surprise_pct}, ...]
        """
        logger.warning("FundamentalCollector.fetch_earnings_dates() not yet implemented.")
        return []
