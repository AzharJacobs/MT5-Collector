"""
Query Utility for MT5 Data
Provides convenient functions to query and analyze collected OHLCV data
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import argparse

from database import DatabaseManager, get_database_summary
from config import TIMEFRAME_ORDER


class DataQuery:
    """Query interface for MT5 OHLCV data"""

    def __init__(self):
        self.db = DatabaseManager()

    def get_candles(
        self,
        timeframe: str,
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        as_dataframe: bool = True
    ):
        """
        Fetch candle data for a specific timeframe.
        Returns pandas DataFrame or list of dicts.
        """
        query = """
        SELECT * FROM ustech_ohlcv
        WHERE timeframe = %s
        """
        params = [timeframe]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if as_dataframe:
                return pd.DataFrame(rows, columns=columns)
            return [dict(zip(columns, row)) for row in rows]

    def get_latest_candles(self, timeframe: str, count: int = 10):
        """Get the most recent candles for a timeframe"""
        return self.get_candles(timeframe, limit=count)

    def get_date_range(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get all candles within a date range"""
        query = """
        SELECT * FROM ustech_ohlcv
        WHERE timeframe = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (timeframe, start_date, end_date))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def get_statistics(self, timeframe: str) -> dict:
        """Get statistical summary for a timeframe"""
        query = """
        SELECT
            COUNT(*) as total_candles,
            MIN(timestamp) as first_candle,
            MAX(timestamp) as last_candle,
            AVG(candle_size) as avg_candle_size,
            AVG(body_size) as avg_body_size,
            AVG(volume) as avg_volume,
            SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_candles,
            SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_candles,
            SUM(CASE WHEN direction = 'neutral' THEN 1 ELSE 0 END) as neutral_candles
        FROM ustech_ohlcv
        WHERE timeframe = %s
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (timeframe,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            return dict(zip(columns, row))

    def get_hourly_distribution(self, timeframe: str) -> pd.DataFrame:
        """Get distribution of candles by hour (useful for ML features)"""
        query = """
        SELECT
            hour,
            COUNT(*) as candle_count,
            AVG(candle_size) as avg_candle_size,
            SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_count,
            SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_count
        FROM ustech_ohlcv
        WHERE timeframe = %s
        GROUP BY hour
        ORDER BY hour
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (timeframe,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def get_daily_distribution(self, timeframe: str) -> pd.DataFrame:
        """Get distribution of candles by day of week"""
        day_order = "CASE day_of_week " + \
                    "WHEN 'Monday' THEN 1 " + \
                    "WHEN 'Tuesday' THEN 2 " + \
                    "WHEN 'Wednesday' THEN 3 " + \
                    "WHEN 'Thursday' THEN 4 " + \
                    "WHEN 'Friday' THEN 5 " + \
                    "WHEN 'Saturday' THEN 6 " + \
                    "WHEN 'Sunday' THEN 7 END"

        query = f"""
        SELECT
            day_of_week,
            COUNT(*) as candle_count,
            AVG(candle_size) as avg_candle_size,
            AVG(volume) as avg_volume
        FROM ustech_ohlcv
        WHERE timeframe = %s
        GROUP BY day_of_week
        ORDER BY {day_order}
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (timeframe,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def export_to_csv(
        self,
        timeframe: str,
        output_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Export data to CSV file. Returns number of rows exported."""
        query = """
        SELECT * FROM ustech_ohlcv
        WHERE timeframe = %s
        """
        params = [timeframe]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(output_path, index=False)
            return len(df)

    def get_ml_dataset(
        self,
        timeframe: str,
        features: List[str] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get a dataset optimized for machine learning.
        Includes all numeric and categorical features.
        """
        if features is None:
            features = [
                'timestamp', 'hour', 'day_of_week', 'month', 'year',
                'open', 'high', 'low', 'close', 'volume',
                'direction', 'candle_size', 'body_size',
                'wick_upper', 'wick_lower'
            ]

        feature_str = ', '.join(features)
        query = f"""
        SELECT {feature_str}
        FROM ustech_ohlcv
        WHERE timeframe = %s
        ORDER BY timestamp ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (timeframe,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)


def print_summary():
    """Print database summary"""
    print("\n" + "=" * 60)
    print("Database Summary")
    print("=" * 60)

    summary = get_database_summary()

    if not summary:
        print("No data found in database.")
        return

    total_candles = 0

    for row in summary:
        tf = row['timeframe']
        count = row['total_candles']
        earliest = row['earliest']
        latest = row['latest']
        total_candles += count

        print(f"\n{tf}:")
        print(f"  Total candles: {count:,}")
        print(f"  Date range: {earliest} to {latest}")

    print(f"\n{'=' * 60}")
    print(f"Total candles across all timeframes: {total_candles:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Query MT5 OHLCV Data')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show database summary'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        help='Timeframe to query (e.g., 1H, 4H, 1D)'
    )
    parser.add_argument(
        '--latest',
        type=int,
        default=10,
        help='Number of latest candles to show'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics for timeframe'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export to CSV file path'
    )

    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.timeframe:
        query = DataQuery()

        if args.stats:
            print(f"\nStatistics for {args.timeframe}:")
            print("-" * 40)
            stats = query.get_statistics(args.timeframe)
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif args.export:
            count = query.export_to_csv(args.timeframe, args.export)
            print(f"Exported {count:,} rows to {args.export}")

        else:
            print(f"\nLatest {args.latest} candles for {args.timeframe}:")
            df = query.get_latest_candles(args.timeframe, args.latest)
            print(df.to_string())

    else:
        print_summary()


if __name__ == "__main__":
    main()
