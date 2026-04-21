"""
Database module for MT5 Data Collector
Handles PostgreSQL connection, schema creation, and data operations

Changes:
  - Added spread column to table and insert
  - Added migration for spread column on existing tables
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5

from config import DB_CONFIG, TIMEFRAME_ORDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseManager:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DB_CONFIG
        self.conn = None

    @contextmanager
    def get_connection(self):
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def create_database(self) -> bool:
        temp_config = self.config.copy()
        temp_config['database'] = 'postgres'

        try:
            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True
            cursor = conn.cursor()

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
            spread INTEGER NOT NULL DEFAULT 0,
            direction TEXT NOT NULL,
            candle_size DECIMAL(18, 6) NOT NULL,
            body_size DECIMAL(18, 6) NOT NULL,
            wick_upper DECIMAL(18, 6) NOT NULL,
            wick_lower DECIMAL(18, 6) NOT NULL,
            session TEXT NOT NULL DEFAULT 'unknown',
            CONSTRAINT unique_symbol_timeframe_timestamp
                UNIQUE (symbol, timeframe, timestamp)
        );
        """
        with self.get_cursor() as cursor:
            cursor.execute(create_table_sql)
            logger.info("Table 'ustech_ohlcv' created/verified successfully")

    def create_index(self) -> None:
        indexes = [
            """
            CREATE INDEX IF NOT EXISTS idx_timeframe_timestamp
            ON ustech_ohlcv (timeframe, timestamp DESC);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_session
            ON ustech_ohlcv (session);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_timeframe_session
            ON ustech_ohlcv (timeframe, session);
            """,
        ]
        with self.get_cursor() as cursor:
            for index_sql in indexes:
                cursor.execute(index_sql)
        logger.info("All indexes created/verified successfully")

    def create_view(self) -> None:
        case_parts = [f"WHEN '{tf}' THEN {order}"
                      for tf, order in TIMEFRAME_ORDER.items()]
        case_statement = "CASE timeframe " + " ".join(case_parts) + " ELSE 99 END"

        # Drop first to avoid column order conflicts when schema changes
        with self.get_cursor() as cursor:
            cursor.execute("DROP VIEW IF EXISTS ustech_view;")
            logger.info("Dropped existing view 'ustech_view'")

        create_view_sql = f"""
        CREATE VIEW ustech_view AS
        SELECT
            id, symbol, timeframe, timestamp, date, time, hour,
            day_of_week, month, year, open, high, low, close,
            volume, spread, direction, candle_size, body_size,
            wick_upper, wick_lower, session
        FROM ustech_ohlcv
        ORDER BY
            {case_statement},
            timestamp DESC;
        """
        with self.get_cursor() as cursor:
            cursor.execute(create_view_sql)
            logger.info("View 'ustech_view' created/replaced successfully")

    def migrate_add_session_column(self) -> None:
        self._add_column_if_missing(
            column='session',
            definition="TEXT NOT NULL DEFAULT 'unknown'"
        )

    def migrate_add_spread_column(self) -> None:
        self._add_column_if_missing(
            column='spread',
            definition="INTEGER NOT NULL DEFAULT 0"
        )

    def _add_column_if_missing(self, column: str, definition: str) -> None:
        check_sql = """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ustech_ohlcv' AND column_name = %s;
        """
        with self.get_cursor() as cursor:
            cursor.execute(check_sql, (column,))
            if not cursor.fetchone():
                cursor.execute(f"ALTER TABLE ustech_ohlcv ADD COLUMN {column} {definition};")
                logger.info(f"Migration complete: '{column}' column added")
            else:
                logger.info(f"Migration skipped: '{column}' column already exists")

    def setup_schema(self) -> None:
        logger.info("Starting database schema setup...")
        self.create_database()
        self.create_table()
        self.migrate_add_session_column()
        self.migrate_add_spread_column()
        self.create_index()
        self.create_view()
        logger.info("Database schema setup completed successfully")

    def insert_candles(self, candles: List[Dict[str, Any]]) -> int:
        if not candles:
            return 0

        insert_sql = """
        INSERT INTO ustech_ohlcv (
            symbol, timeframe, timestamp, date, time, hour,
            day_of_week, month, year, open, high, low, close,
            volume, spread, direction, candle_size, body_size,
            wick_upper, wick_lower, session
        ) VALUES %s
        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING;
        """

        values = [
            (
                c['symbol'], c['timeframe'], c['timestamp'], c['date'],
                c['time'], c['hour'], c['day_of_week'], c['month'],
                c['year'], c['open'], c['high'], c['low'], c['close'],
                c['volume'], c.get('spread', 0), c['direction'],
                c['candle_size'], c['body_size'], c['wick_upper'],
                c['wick_lower'], c['session']
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
        query = """
        SELECT MAX(timestamp) FROM ustech_ohlcv
        WHERE symbol = %s AND timeframe = %s;
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (symbol, timeframe))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_row_count(self, timeframe: str = None) -> int:
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
                WHEN '1min'  THEN 1  WHEN '2min'  THEN 2
                WHEN '3min'  THEN 3  WHEN '4min'  THEN 4
                WHEN '5min'  THEN 5  WHEN '10min' THEN 6
                WHEN '15min' THEN 7  WHEN '30min' THEN 8
                WHEN '1H'    THEN 9  WHEN '4H'    THEN 10
                WHEN '1D'    THEN 11 ELSE 99
            END;
        """
        with self.get_cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_session_summary(self, timeframe: str = None) -> List[Dict[str, Any]]:
        if timeframe:
            query = """
            SELECT session, COUNT(*) as total_candles,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
            FROM ustech_ohlcv WHERE timeframe = %s
            GROUP BY session ORDER BY total_candles DESC;
            """
            params = (timeframe,)
        else:
            query = """
            SELECT session, COUNT(*) as total_candles,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
            FROM ustech_ohlcv
            GROUP BY session ORDER BY total_candles DESC;
            """
            params = None

        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def setup_database():
    db = DatabaseManager()
    db.setup_schema()
    return db


def get_database_summary():
    db = DatabaseManager()
    return db.get_summary()


if __name__ == "__main__":
    print("Setting up database schema...")
    setup_database()
    print("Done!")