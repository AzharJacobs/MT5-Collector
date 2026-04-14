"""
Configuration module for MT5 Data Collector
Loads environment variables and defines constants
"""

import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "ustech_ohlcv"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# MetaTrader 5
MT5_CONFIG = {
    "login":    int(os.getenv("MT5_LOGIN", 0)) if os.getenv("MT5_LOGIN") else None,
    "password": os.getenv("MT5_PASSWORD"),
    "server":   os.getenv("MT5_SERVER"),
}

# Symbol
SYMBOL = os.getenv("SYMBOL", "USTECm")

# Chunk size (used by DB batch inserts)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 10000))

# Timeframes — display name → MT5 constant name
TIMEFRAMES = [
    ("1min",  "TIMEFRAME_M1"),
    ("2min",  "TIMEFRAME_M2"),
    ("3min",  "TIMEFRAME_M3"),
    ("4min",  "TIMEFRAME_M4"),
    ("5min",  "TIMEFRAME_M5"),
    ("10min", "TIMEFRAME_M10"),
    ("15min", "TIMEFRAME_M15"),
    ("30min", "TIMEFRAME_M30"),
    ("1H",    "TIMEFRAME_H1"),
    ("4H",    "TIMEFRAME_H4"),
    ("1D",    "TIMEFRAME_D1"),
]

# Date range: full year 2024 up to Jan 1 2025
DATA_START_DATE = os.getenv("DATA_START_DATE", "2024-01-01")
DATA_END_DATE   = os.getenv("DATA_END_DATE",   "2025-01-01")

# Timeframe display order for DB views
TIMEFRAME_ORDER = {
    "1min": 1, "2min": 2, "3min": 3, "4min": 4,
    "5min": 5, "10min": 6, "15min": 7, "30min": 8,
    "1H": 9, "4H": 10, "1D": 11,
}