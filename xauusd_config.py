"""
Configuration module for XAUUSD MT5 Data Collector
Loads environment variables and defines constants
"""

import os
from dotenv import load_dotenv

load_dotenv("xauusd.env", override=True)

# PostgreSQL
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "XAUUSD"),
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
SYMBOL = os.getenv("SYMBOL", "XAUUSDm")

# Chunk size (used by DB batch inserts)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 10000))

# Timeframes — display name → MT5 constant name
TIMEFRAMES = [
    ("5min",  "TIMEFRAME_M5"),
    ("15min", "TIMEFRAME_M15"),
    ("1H",    "TIMEFRAME_H1"),
    ("4H",    "TIMEFRAME_H4"),
]

# Date range
DATA_START_DATE = os.getenv("DATA_START_DATE", "2022-01-01")
DATA_END_DATE   = os.getenv("DATA_END_DATE",   "2026-01-01")

# Timeframe display order for DB views
TIMEFRAME_ORDER = {
    "5min": 1, "15min": 2, "1H": 3, "4H": 4,
}
