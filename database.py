"""
Database module for MT5 Data Collector
Handles PostgreSQL connection, schema creation, and data operations
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging

from config import DB_CONFIG, TIMEFRAME_ORDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DB_CONFIG
        self.conn = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def create_database(self) -> bool:
        """
        Create the ustech_data database if it doesn't exist.
        Returns True if created, False if already exists.
        """
        # Connect to default 'postgres' database to create our database
        temp_config = self.config.copy()
        temp_config['database'] = 'postgres'

        try:
            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.config['database'],)
            )

            if cursor.fetchone() is None:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(self.config['database'])
                    )
                )
                logger.info(f"Database '{self.config['database']}' created successfully")
                created = True
            else:
                logger.info(f"Database '{self.config['database']}' already exists")
                created = False

            cursor.close()
            conn.close()
            return created

        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise

    def create_table(self) -> None:
        """Create the ustech_ohlcv table with all required columns"""

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ustech_ohlcv (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week TEXT NOT NULL,
            month INTEGER NOT NULL,
            year INTEGER NOT NULL,
            open DECIMAL(18, 6) NOT NULL,
            high DECIMAL(18, 6) NOT NULL,
            low DECIMAL(18, 6) NOT NULL,
            close DECIMAL(18, 6) NOT NULL,
            volume DECIMAL(18, 6) NOT NULL,
            direction TEXT NOT NULL,
            candle_size DECIMAL(18, 6) NOT NULL,
            body_size DECIMAL(18, 6) NOT NULL,
            wick_upper DECIMAL(18, 6) NOT NULL,
            wick_lower DECIMAL(18, 6) NOT NULL,

            -- Unique constraint to prevent duplicates
            CONSTRAINT unique_symbol_timeframe_timestamp
                UNIQUE (symbol, timeframe, timestamp)
        );
        """

        with self.get_cursor() as cursor:
            cursor.execute(create_table_sql)
            logger.info("Table 'ustech_ohlcv' created/verified successfully")

    def create_index(self) -> None:
        """Create index on (timeframe, timestamp DESC) for fast queries"""

        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_timeframe_timestamp
        ON ustech_ohlcv (timeframe, timestamp DESC);
        """

        with self.get_cursor() as cursor:
            cursor.execute(create_index_sql)
            logger.info("Index 'idx_timeframe_timestamp' created/verified successfully")

    def create_view(self) -> None:
        """
        Create ustech_view that displays data grouped by timeframe
        in the specified order, sorted newest to oldest within each block
        """

        # Build CASE statement for timeframe ordering
        case_parts = [f"WHEN '{tf}' THEN {order}"
                      for tf, order in TIMEFRAME_ORDER.items()]
        case_statement = "CASE timeframe " + " ".join(case_parts) + " ELSE 99 END"

        create_view_sql = f"""
        CREATE OR REPLACE VIEW ustech_view AS
        SELECT
            id,
            symbol,
            timeframe,
            timestamp,
            date,
            time,
            hour,
            day_of_week,
            month,
            year,
            open,
            high,
            low,
            close,
            volume,
            direction,
            candle_size,
            body_size,
            wick_upper,
            wick_lower
        FROM ustech_ohlcv
        ORDER BY
            {case_statement},
            timestamp DESC;
        """

        with self.get_cursor() as cursor:
            cursor.execute(create_view_sql)
            logger.info("View 'ustech_view' created/replaced successfully")

    def setup_schema(self) -> None:
        """Complete schema setup: database, table, index, and view"""
        logger.info("Starting database schema setup...")
        self.create_database()
        self.create_table()
        self.create_index()
        self.create_view()
        logger.info("Database schema setup completed successfully")

    def insert_candles(self, candles: List[Dict[str, Any]]) -> int:
        """
        Insert candle data into the database.
        Uses ON CONFLICT DO NOTHING to handle duplicates gracefully.
        Returns the number of rows inserted.
        """
        if not candles:
            return 0

        insert_sql = """
        INSERT INTO ustech_ohlcv (
            symbol, timeframe, timestamp, date, time, hour,
            day_of_week, month, year, open, high, low, close,
            volume, direction, candle_size, body_size, wick_upper, wick_lower
        ) VALUES %s
        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING;
        """

        # Prepare values as tuples
        values = [
            (
                c['symbol'], c['timeframe'], c['timestamp'], c['date'],
                c['time'], c['hour'], c['day_of_week'], c['month'],
                c['year'], c['open'], c['high'], c['low'], c['close'],
                c['volume'], c['direction'], c['candle_size'],
                c['body_size'], c['wick_upper'], c['wick_lower']
            )
            for c in candles
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            execute_values(cursor, insert_sql, values)
            inserted = cursor.rowcount
            cursor.close()
            return inserted

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """Get the latest timestamp for a symbol/timeframe combination"""

        query = """
        SELECT MAX(timestamp) FROM ustech_ohlcv
        WHERE symbol = %s AND timeframe = %s;
        """

        with self.get_cursor() as cursor:
            cursor.execute(query, (symbol, timeframe))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_row_count(self, timeframe: str = None) -> int:
        """Get total row count, optionally filtered by timeframe"""

        if timeframe:
            query = "SELECT COUNT(*) FROM ustech_ohlcv WHERE timeframe = %s;"
            params = (timeframe,)
        else:
            query = "SELECT COUNT(*) FROM ustech_ohlcv;"
            params = None

        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()[0]

    def get_summary(self) -> List[Dict[str, Any]]:
        """Get summary statistics grouped by timeframe"""

        query = """
        SELECT
            timeframe,
            COUNT(*) as total_candles,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest
        FROM ustech_ohlcv
        GROUP BY timeframe
        ORDER BY
            CASE timeframe
                WHEN '1min' THEN 1
                WHEN '2min' THEN 2
                WHEN '3min' THEN 3
                WHEN '4min' THEN 4
                WHEN '5min' THEN 5
                WHEN '10min' THEN 6
                WHEN '15min' THEN 7
                WHEN '30min' THEN 8
                WHEN '1H' THEN 9
                WHEN '4H' THEN 10
                WHEN '1D' THEN 11
                ELSE 99
            END;
        """

        with self.get_cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# Standalone functions for quick access
def setup_database():
    """Quick function to setup the complete database schema"""
    db = DatabaseManager()
    db.setup_schema()
    return db


def get_database_summary():
    """Quick function to get database summary"""
    db = DatabaseManager()
    return db.get_summary()


if __name__ == "__main__":
    # When run directly, setup the database
    print("Setting up database schema...")
    setup_database()
    print("Done!")
