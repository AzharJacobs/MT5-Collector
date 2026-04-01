MT5 OHLCV Data Collector
A complete Python + PostgreSQL data collection system for MetaTrader 5. Collects historical OHLCV chart data and stores it in a structured, ML-ready PostgreSQL database.

Features
Multi-timeframe support: 1min, 2min, 3min, 4min, 5min, 10min, 15min, 30min, 1H, 4H, 1D
Chunked data fetching: Handles large history without hitting MT5 limits
Derived columns: Pre-calculated features for ML (direction, candle_size, body_size, wicks, etc.)
Duplicate handling: Safe to run multiple times without data corruption
Incremental updates: Fetch only new data since last run
ML-ready output: Database view with proper ordering and export utilities
Production logging: Rotating file logs with configurable levels
Data validation: Quality checks on OHLCV data before insertion
Scheduled automation: Windows Task Scheduler integration
Requirements
Windows OS (MT5 requirement)
MetaTrader 5 installed and configured
PostgreSQL 16
Python 3.8+
Installation
1. Clone/Download the project
cd mt5-data-collector
2. Create a virtual environment (recommended)
python -m venv venv

venv\Scripts\activate  # Windows
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables
Copy the example environment file and edit it:

copy .env.example .env
Edit .env with your settings:

# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ustech_data
DB_USER=postgres
DB_PASSWORD=your_password

# MetaTrader 5 Configuration (optional if MT5 is already logged in)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Data Collection Settings
SYMBOL=USTech
CHUNK_SIZE=10000
5. Ensure PostgreSQL is running
Make sure your PostgreSQL server is running and accessible.

Usage
Collect Data (Main Script)
# Incremental update (default - fetches only new data)

python mt5_collector.py


# Full history fetch (fetches all available data)

python mt5_collector.py --full


# Skip database setup (use if schema already exists)

python mt5_collector.py --skip-db-setup


# Use a different symbol

python mt5_collector.py --symbol "US100"
Setup Database Only
python database.py
Query Data
# Show database summary

python query_data.py --summary


# Get latest candles for a timeframe

python query_data.py --timeframe 1H --latest 20


# Get statistics for a timeframe

python query_data.py --timeframe 1D --stats


# Export to CSV

python query_data.py --timeframe 1H --export hourly_data.csv
Database Schema
Table: ustech_ohlcv
Column	Type	Description
id	SERIAL	Auto-increment primary key
symbol	TEXT	Trading symbol (e.g., "USTech")
timeframe	TEXT	Timeframe (e.g., "1H", "4H", "1D")
timestamp	TIMESTAMP	Full datetime of the candle
date	DATE	Date portion only
time	TIME	Time portion only
hour	INTEGER	Hour (0-23)
day_of_week	TEXT	Day name (e.g., "Monday")
month	INTEGER	Month (1-12)
year	INTEGER	Year
open	DECIMAL(18,6)	Open price
high	DECIMAL(18,6)	High price
low	DECIMAL(18,6)	Low price
close	DECIMAL(18,6)	Close price
volume	DECIMAL(18,6)	Tick volume
direction	TEXT	"buy", "sell", or "neutral"
candle_size	DECIMAL(18,6)	high - low
body_size	DECIMAL(18,6)	abs(close - open)
wick_upper	DECIMAL(18,6)	high - max(open, close)
wick_lower	DECIMAL(18,6)	min(open, close) - low
Index
idx_timeframe_timestamp: Composite index on (timeframe, timestamp DESC) for fast queries
View: ustech_view
A view that displays all data grouped by timeframe in order:

1min → 2min → 3min → 4min → 5min → 10min → 15min → 30min → 1H → 4H → 1D
Within each timeframe block, data is sorted newest to oldest.

SELECT * FROM ustech_view LIMIT 100;
Python API
Using the Collector
from mt5_collector import MT5Collector


# Create collector

collector = MT5Collector(symbol="USTech")


# Run collection (setup DB, init MT5, fetch all timeframes)

results = collector.run(setup_db=True, incremental=True)


print(results['success'])  # True/False

print(results['timeframes'])  # Stats per timeframe
Querying Data
from query_data import DataQuery

import pandas as pd


query = DataQuery()


# Get latest candles

df = query.get_latest_candles("1H", count=100)


# Get date range

from datetime import datetime

df = query.get_date_range(

    "1D",

    start_date=datetime(2024, 1, 1),

    end_date=datetime(2024, 12, 31)

)


# Get ML-ready dataset

df = query.get_ml_dataset("4H", limit=10000)


# Get statistics

stats = query.get_statistics("1H")


# Get hourly distribution (for feature engineering)

hourly = query.get_hourly_distribution("15min")


# Export to CSV

query.export_to_csv("1D", "daily_data.csv")
Direction Logic
The direction column is calculated as:

"buy": When close > open (bullish candle)
"sell": When close < open (bearish candle)
"neutral": When close == open (doji candle)
Data Validation
The validator performs comprehensive quality checks before database insertion:

Check	Description
Required fields	All columns must be present and non-null
Data types	Numeric, integer, and string type validation
OHLCV logic	High >= Low, High >= Open/Close, Low <= Open/Close
Price range	Prices within configurable min/max bounds
Volume	Non-negative, warning for zero or unusually high
Timestamp	Not in future, not before 1990
Derived fields	Candle size, body size, wick calculations verified
Outliers	Statistical outlier detection (configurable threshold)
Disable Validation
For faster processing (not recommended for production):

python mt5_collector.py --no-validation
Production Logging
Logs are stored in the logs/ directory with automatic rotation:

Log File	Description
mt5_collector.log	Main log with size-based rotation (10MB, 10 backups)
mt5_daily.log	Daily rotation for historical analysis
error.log	Errors only for quick debugging
Log Levels
DEBUG: Detailed chunk processing, validation warnings
INFO: Collection progress, timeframe summaries
WARNING: Invalid records, large candles, outliers
ERROR: Connection failures, critical errors
Task Scheduler (Automation)
Use the scheduler module to automate data collection:

Open Task Scheduler
Create Basic Task
Set trigger (e.g., every hour)
Action: Start a program
Program: python
Arguments: C:\path\to\mt5-data-collector\mt5_collector.py
Start in: C:\path\to\mt5-data-collector
Troubleshooting
Symbol not found
If you get "Symbol 'USTech' not found", check available symbols:

import MetaTrader5 as mt5

mt5.initialize()

symbols = mt5.symbols_get()

for s in symbols:

    if 'tech' in s.name.lower() or 'nas' in s.name.lower():

        print(s.name)

mt5.shutdown()
Common alternatives: US100, USTEC, NAS100, NASDAQ100

MT5 not initialized
Ensure:

MetaTrader 5 is installed
MT5 is running and logged in
The account has access to the symbol
PostgreSQL connection failed
Ensure:

PostgreSQL service is running
Credentials in .env are correct
Database user has CREATE DATABASE permissions
File Structure
mt5-data-collector/
├── .env.example      # Environment template
├── .env              # Your configuration (create this)
├── requirements.txt  # Python dependencies
├── config.py         # Configuration loader
├── database.py       # Database operations
├── mt5_collector.py  # Main data collector
├── query_data.py     # Query utilities
└── README.md         # This file
License
MIT License - Free for personal and commercial use.