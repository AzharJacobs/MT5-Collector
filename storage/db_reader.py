"""
DB Reader
Shared read interface for pulling market data from the database.
Used by the ML engine, feature engineering, and dataset builders.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from storage.db_writer import DatabaseManager
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataReader(DatabaseManager):
    """
    Read-only interface over the OHLCV database.
    Extends DatabaseManager with query utilities for ML pipelines.
    """

    def get_candles(
        self,
        timeframe: str,
        symbol: str = None,
        limit: int = None,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict[str, Any]]:
        """Fetch candles with optional filters. Returns list of dicts."""
        conditions = ["timeframe = %s"]
        params: list = [timeframe]

        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        if start_date:
            conditions.append("timestamp >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= %s")
            params.append(end_date)

        where  = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""
        query  = f"""
        SELECT * FROM ustech_ohlcv
        WHERE {where}
        ORDER BY timestamp ASC
        {limit_clause};
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_candles_df(
        self,
        timeframe: str,
        symbol: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Return candles as a pandas DataFrame (for ML pipelines)."""
        from sqlalchemy import create_engine
        engine = create_engine(
            f"postgresql://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
        conditions = ["timeframe = %(timeframe)s"]
        params: dict = {"timeframe": timeframe}
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params["symbol"] = symbol
        if start_date:
            conditions.append("timestamp >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            conditions.append("timestamp <= %(end_date)s")
            params["end_date"] = end_date

        where = " AND ".join(conditions)
        query = f"SELECT * FROM ustech_ohlcv WHERE {where} ORDER BY timestamp ASC;"
        return pd.read_sql(query, engine, params=params)

    def get_latest_candles(
        self,
        timeframe: str,
        n: int = 100,
        symbol: str = None,
    ) -> List[Dict[str, Any]]:
        """Get the N most recent candles for a timeframe."""
        conditions = ["timeframe = %s"]
        params: list = [timeframe]
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        where = " AND ".join(conditions)
        query = f"""
        SELECT * FROM ustech_ohlcv
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT %s;
        """
        params.append(n)
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_date_range(self, timeframe: str) -> Dict[str, Any]:
        """Return earliest and latest timestamps for a timeframe."""
        query = """
        SELECT MIN(timestamp) AS earliest, MAX(timestamp) AS latest,
               COUNT(*) AS total_candles
        FROM ustech_ohlcv WHERE timeframe = %s;
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (timeframe,))
            row = cursor.fetchone()
            return {
                "timeframe":     timeframe,
                "earliest":      row[0],
                "latest":        row[1],
                "total_candles": row[2],
            }

    def get_hourly_distribution(self, timeframe: str) -> List[Dict[str, Any]]:
        """Return candle count grouped by hour of day."""
        query = """
        SELECT hour, COUNT(*) AS count,
               ROUND(AVG(close - open), 4) AS avg_move,
               ROUND(AVG(candle_size), 4) AS avg_range
        FROM ustech_ohlcv WHERE timeframe = %s
        GROUP BY hour ORDER BY hour;
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (timeframe,))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def export_to_csv(
        self,
        timeframe: str,
        filepath: str,
        start_date: str = None,
        end_date: str = None,
    ) -> int:
        """Export candles to CSV. Returns number of rows written."""
        df = self.get_candles_df(timeframe, start_date=start_date, end_date=end_date)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df):,} rows ({timeframe}) → {filepath}")
        return len(df)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def get_reader(config: Dict[str, Any] = None) -> DataReader:
    return DataReader(config or DB_CONFIG)


if __name__ == "__main__":
    reader = get_reader()
    for row in reader.get_summary():
        print(f"  {row['timeframe']:8s}: {row['total_candles']:>10,} candles | "
              f"{row['earliest']} → {row['latest']}")
