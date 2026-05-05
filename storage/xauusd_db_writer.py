"""
XAUUSD DB Writer
Manages the PostgreSQL connection and all write/schema operations.
Clean validated data comes in here and gets persisted.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from contextlib import contextmanager
from datetime import datetime as _dt
from typing import List, Dict, Any, Optional
import logging

from xauusd_config import DB_CONFIG, TIMEFRAME_ORDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseManager:
    """PostgreSQL connection manager and schema/write operations."""

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

    # -------------------------------------------------------------------------
    # Schema setup
    # -------------------------------------------------------------------------
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
                logger.info(f"Database '{self.config['database']}' created")
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
        CREATE TABLE IF NOT EXISTS xauusd_ohlcv (
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
            logger.info("Table 'xauusd_ohlcv' created/verified")

    def create_verified_table(self) -> None:
        """Create xauusd_verified — the post-validation table read by the ML engine."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS xauusd_verified (
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
            gap_checked BOOLEAN NOT NULL DEFAULT FALSE,
            duplicate_checked BOOLEAN NOT NULL DEFAULT FALSE,
            anomaly_checked BOOLEAN NOT NULL DEFAULT FALSE,
            is_verified BOOLEAN NOT NULL DEFAULT FALSE,
            verified_at TIMESTAMP,
            source_id INTEGER,
            CONSTRAINT unique_verified_symbol_timeframe_timestamp
                UNIQUE (symbol, timeframe, timestamp)
        );
        """
        with self.get_cursor() as cursor:
            cursor.execute(create_table_sql)
            logger.info("Table 'xauusd_verified' created/verified")

    def create_index(self) -> None:
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_xauusd_timeframe_timestamp ON xauusd_ohlcv (timeframe, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_xauusd_session ON xauusd_ohlcv (session);",
            "CREATE INDEX IF NOT EXISTS idx_xauusd_timeframe_session ON xauusd_ohlcv (timeframe, session);",
        ]
        with self.get_cursor() as cursor:
            for idx in indexes:
                cursor.execute(idx)
        logger.info("All indexes created/verified")

    def create_view(self) -> None:
        case_parts = [f"WHEN '{tf}' THEN {order}" for tf, order in TIMEFRAME_ORDER.items()]
        case_stmt  = "CASE timeframe " + " ".join(case_parts) + " ELSE 99 END"

        with self.get_cursor() as cursor:
            cursor.execute("DROP VIEW IF EXISTS xauusd_view;")

        create_view_sql = f"""
        CREATE VIEW xauusd_view AS
        SELECT
            id, symbol, timeframe, timestamp, date, time, hour,
            day_of_week, month, year, open, high, low, close,
            volume, spread, direction, candle_size, body_size,
            wick_upper, wick_lower, session
        FROM xauusd_ohlcv
        ORDER BY {case_stmt}, timestamp DESC;
        """
        with self.get_cursor() as cursor:
            cursor.execute(create_view_sql)
        logger.info("View 'xauusd_view' created/replaced")

    def migrate_add_session_column(self) -> None:
        self._add_column_if_missing('session', "TEXT NOT NULL DEFAULT 'unknown'")

    def migrate_add_spread_column(self) -> None:
        self._add_column_if_missing('spread', "INTEGER NOT NULL DEFAULT 0")

    def migrate_add_unique_constraint(self) -> None:
        check_sql = """
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'xauusd_ohlcv'::regclass
          AND conname = 'unique_symbol_timeframe_timestamp';
        """
        add_sql = """
        ALTER TABLE xauusd_ohlcv
        ADD CONSTRAINT unique_symbol_timeframe_timestamp
        UNIQUE (symbol, timeframe, timestamp);
        """
        with self.get_cursor() as cursor:
            cursor.execute(check_sql)
            if not cursor.fetchone():
                cursor.execute(add_sql)
                logger.info("Migration: unique constraint on (symbol, timeframe, timestamp) added")
            else:
                logger.debug("Migration skipped: unique constraint already exists")

    def migrate_add_missing_columns(self) -> None:
        """Add any columns that may be absent on pre-existing tables."""
        migrations = [
            ('symbol',      "TEXT NOT NULL DEFAULT 'XAUUSD'"),
            ('timeframe',   "TEXT NOT NULL DEFAULT ''"),
            ('date',        "DATE"),
            ('time',        "TIME"),
            ('hour',        "INTEGER"),
            ('day_of_week', "TEXT NOT NULL DEFAULT ''"),
            ('month',       "INTEGER"),
            ('year',        "INTEGER"),
            ('spread',      "INTEGER NOT NULL DEFAULT 0"),
            ('direction',   "TEXT NOT NULL DEFAULT ''"),
            ('candle_size', "DECIMAL(18, 6)"),
            ('body_size',   "DECIMAL(18, 6)"),
            ('wick_upper',  "DECIMAL(18, 6)"),
            ('wick_lower',  "DECIMAL(18, 6)"),
            ('session',     "TEXT NOT NULL DEFAULT 'unknown'"),
        ]
        for column, definition in migrations:
            self._add_column_if_missing(column, definition)

    def _add_column_if_missing(self, column: str, definition: str) -> None:
        check_sql = """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'xauusd_ohlcv' AND column_name = %s;
        """
        with self.get_cursor() as cursor:
            cursor.execute(check_sql, (column,))
            if not cursor.fetchone():
                cursor.execute(f"ALTER TABLE xauusd_ohlcv ADD COLUMN {column} {definition};")
                logger.info(f"Migration: '{column}' column added")
            else:
                logger.debug(f"Migration skipped: '{column}' already exists")

    def setup_schema(self) -> None:
        logger.info("Starting database schema setup...")
        self.create_database()
        self.create_table()
        self.migrate_add_missing_columns()
        self.migrate_add_unique_constraint()
        self.create_verified_table()
        self.create_index()
        self.create_view()
        logger.info("Schema setup complete")

    # -------------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------------
    def insert_candles(self, candles: List[Dict[str, Any]]) -> int:
        if not candles:
            return 0

        insert_sql = """
        INSERT INTO xauusd_ohlcv (
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

    def insert_verified_candles(self, candles: List[Dict[str, Any]]) -> int:
        """Insert validated candles into xauusd_verified with all check flags TRUE."""
        if not candles:
            return 0

        verified_at = _dt.utcnow()
        insert_sql = """
        INSERT INTO xauusd_verified (
            symbol, timeframe, timestamp, date, time, hour,
            day_of_week, month, year, open, high, low, close,
            volume, spread, direction, candle_size, body_size,
            wick_upper, wick_lower, session,
            gap_checked, duplicate_checked, anomaly_checked,
            is_verified, verified_at, source_id
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
                c['wick_lower'], c['session'],
                True, True, True, True, verified_at, c.get('id'),
            )
            for c in candles
        ]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            execute_values(cursor, insert_sql, values)
            inserted = cursor.rowcount
            cursor.close()
            return inserted

    # -------------------------------------------------------------------------
    # Read helpers (used internally by writer and by db_reader)
    # -------------------------------------------------------------------------
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        query = "SELECT MAX(timestamp) FROM xauusd_ohlcv WHERE symbol = %s AND timeframe = %s;"
        with self.get_cursor() as cursor:
            cursor.execute(query, (symbol, timeframe))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_earliest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        query = "SELECT MIN(timestamp) FROM xauusd_ohlcv WHERE symbol = %s AND timeframe = %s;"
        with self.get_cursor() as cursor:
            cursor.execute(query, (symbol, timeframe))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_row_count(self, timeframe: str = None) -> int:
        if timeframe:
            query  = "SELECT COUNT(*) FROM xauusd_ohlcv WHERE timeframe = %s;"
            params = (timeframe,)
        else:
            query  = "SELECT COUNT(*) FROM xauusd_ohlcv;"
            params = None
        with self.get_cursor() as cursor:
            cursor.execute(query, params) if params else cursor.execute(query)
            return cursor.fetchone()[0]

    def get_summary(self) -> List[Dict[str, Any]]:
        query = """
        SELECT timeframe, COUNT(*) AS total_candles,
               MIN(timestamp) AS earliest, MAX(timestamp) AS latest
        FROM xauusd_ohlcv
        GROUP BY timeframe
        ORDER BY
            CASE timeframe
                WHEN '5min' THEN 5 WHEN '15min' THEN 7
                WHEN '1H'   THEN 9 WHEN '4H'    THEN 10
                ELSE 99
            END;
        """
        with self.get_cursor() as cursor:
            cursor.execute(query)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_session_summary(self, timeframe: str = None) -> List[Dict[str, Any]]:
        if timeframe:
            query  = """
            SELECT session, COUNT(*) AS total_candles,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM xauusd_ohlcv WHERE timeframe = %s
            GROUP BY session ORDER BY total_candles DESC;
            """
            params = (timeframe,)
        else:
            query  = """
            SELECT session, COUNT(*) AS total_candles,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM xauusd_ohlcv
            GROUP BY session ORDER BY total_candles DESC;
            """
            params = None
        with self.get_cursor() as cursor:
            cursor.execute(query, params) if params else cursor.execute(query)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def setup_database() -> DatabaseManager:
    db = DatabaseManager()
    db.setup_schema()
    return db


def get_database_summary() -> List[Dict[str, Any]]:
    return DatabaseManager().get_summary()


if __name__ == "__main__":
    print("Setting up XAUUSD database schema...")
    setup_database()
    print("Done!")
