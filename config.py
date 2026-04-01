"""
Configuration module for MT5 Data Collector
Loads environment variables and defines constants
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "ustech_data"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# MetaTrader 5 Configuration
MT5_CONFIG = {
    "login": int(os.getenv("MT5_LOGIN", 0)) if os.getenv("MT5_LOGIN") else None,
    "password": os.getenv("MT5_PASSWORD"),
    "server": os.getenv("MT5_SERVER"),
}

# Symbol to collect data for
SYMBOL = os.getenv("SYMBOL", "USTech")

# Chunk size for fetching historical data
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 10000))

# Timeframe mapping: display name -> MT5 timeframe constant
# MT5 timeframe constants will be imported in the main module
TIMEFRAMES = [
    ("1min", "TIMEFRAME_M1"),
    ("2min", "TIMEFRAME_M2"),
    ("3min", "TIMEFRAME_M3"),
    ("4min", "TIMEFRAME_M4"),
    ("5min", "TIMEFRAME_M5"),
    ("10min", "TIMEFRAME_M10"),
    ("15min", "TIMEFRAME_M15"),
    ("30min", "TIMEFRAME_M30"),
    ("1H", "TIMEFRAME_H1"),
    ("4H", "TIMEFRAME_H4"),
    ("1D", "TIMEFRAME_D1"),
]

# Timeframe order for the view (index for sorting)
TIMEFRAME_ORDER = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "4min": 4,
    "5min": 5,
    "10min": 6,
    "15min": 7,
    "30min": 8,
    "1H": 9,
    "4H": 10,
    "1D": 11,
}
