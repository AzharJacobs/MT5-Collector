# MT5 OHLCV Data Collector

> A Python + PostgreSQL system for collecting, validating, and storing MetaTrader 5 historical OHLCV data — structured for ML-ready use.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Python API](#python-api)
- [Data Validation](#data-validation)
- [Logging](#logging)
- [Automation](#automation)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [License](#license)

---

## Features

| Feature | Description |
|---|---|
| **Multi-timeframe** | 1min, 2min, 3min, 4min, 5min, 10min, 15min, 30min, 1H, 4H, 1D |
| **Chunked fetching** | Handles large history without hitting MT5 API limits |
| **Derived columns** | Pre-calculated ML features: direction, candle size, body size, wicks |
| **Duplicate handling** | Safe to re-run — no data corruption |
| **Incremental updates** | Only fetches new data since the last run |
| **Data validation** | Quality checks on all OHLCV data before insertion |
| **Production logging** | Rotating file logs with configurable levels |
| **Automation** | Windows Task Scheduler integration |

---

## Requirements

- **OS**: Windows (MT5 requirement)
- **MetaTrader 5**: Installed and configured
- **PostgreSQL**: v16+
- **Python**: 3.8+

---

## Installation

**1. Clone or download the project**

```bash
cd mt5-data-collector
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy the example environment file and fill in your values:

```bash
copy .env.example .env
```

```env
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ustech_data
DB_USER=postgres
DB_PASSWORD=your_password

# MetaTrader 5 (optional if MT5 is already logged in)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Data Collection
SYMBOL=USTech
CHUNK_SIZE=10000
```

Ensure your PostgreSQL server is running before proceeding.

---

## Usage

### Collect Data

```bash
# Incremental update — fetches only new data (default)
python mt5_collector.py

# Full history fetch
python mt5_collector.py --full

# Skip DB setup (if schema already exists)
python mt5_collector.py --skip-db-setup

# Use a different symbol
python mt5_collector.py --symbol "US100"

# Disable validation (faster, not recommended for production)
python mt5_collector.py --no-validation
```

### Setup Database Only

```bash
python database.py
```

### Query Data

```bash
# Show database summary
python query_data.py --summary

# Get latest candles for a timeframe
python query_data.py --timeframe 1H --latest 20

# Get statistics for a timeframe
python query_data.py --timeframe 1D --stats

# Export to CSV
python query_data.py --timeframe 1H --export hourly_data.csv
```

---

## Database Schema

### Table: `ustech_ohlcv`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Auto-increment primary key |
| `symbol` | TEXT | Trading symbol (e.g. `"USTech"`) |
| `timeframe` | TEXT | Timeframe (e.g. `"1H"`, `"4H"`, `"1D"`) |
| `timestamp` | TIMESTAMP | Full datetime of the candle |
| `date` | DATE | Date portion only |
| `time` | TIME | Time portion only |
| `hour` | INTEGER | Hour (0–23) |
| `day_of_week` | TEXT | Day name (e.g. `"Monday"`) |
| `month` | INTEGER | Month (1–12) |
| `year` | INTEGER | Year |
| `open` | DECIMAL(18,6) | Open price |
| `high` | DECIMAL(18,6) | High price |
| `low` | DECIMAL(18,6) | Low price |
| `close` | DECIMAL(18,6) | Close price |
| `volume` | DECIMAL(18,6) | Tick volume |
| `direction` | TEXT | `"buy"`, `"sell"`, or `"neutral"` |
| `candle_size` | DECIMAL(18,6) | `high - low` |
| `body_size` | DECIMAL(18,6) | `abs(close - open)` |
| `wick_upper` | DECIMAL(18,6) | `high - max(open, close)` |
| `wick_lower` | DECIMAL(18,6) | `min(open, close) - low` |

**Index:** `idx_timeframe_timestamp` — composite index on `(timeframe, timestamp DESC)` for fast queries.

### Direction Logic

| Value | Condition |
|---|---|
| `"buy"` | `close > open` (bullish candle) |
| `"sell"` | `close < open` (bearish candle) |
| `"neutral"` | `close == open` (doji candle) |

### View: `ustech_view`

Displays all data grouped by timeframe in order, with each block sorted newest to oldest:

```
1min → 2min → 3min → 4min → 5min → 10min → 15min → 30min → 1H → 4H → 1D
```

```sql
SELECT * FROM ustech_view LIMIT 100;
```

---

## Python API

### Running the Collector

```python
from mt5_collector import MT5Collector

collector = MT5Collector(symbol="USTech")
results = collector.run(setup_db=True, incremental=True)

print(results['success'])     # True/False
print(results['timeframes'])  # Stats per timeframe
```

### Querying Data

```python
from query_data import DataQuery
from datetime import datetime

query = DataQuery()

# Latest candles
df = query.get_latest_candles("1H", count=100)

# Date range
df = query.get_date_range("1D", start_date=datetime(2024, 1, 1), end_date=datetime(2024, 12, 31))

# ML-ready dataset
df = query.get_ml_dataset("4H", limit=10000)

# Statistics
stats = query.get_statistics("1H")

# Hourly distribution (for feature engineering)
hourly = query.get_hourly_distribution("15min")

# Export to CSV
query.export_to_csv("1D", "daily_data.csv")
```

---

## Data Validation

The validator performs quality checks before every database insertion:

| Check | Description |
|---|---|
| Required fields | All columns must be present and non-null |
| Data types | Numeric, integer, and string type validation |
| OHLCV logic | `High >= Low`, `High >= Open/Close`, `Low <= Open/Close` |
| Price range | Prices within configurable min/max bounds |
| Volume | Non-negative; warnings for zero or unusually high values |
| Timestamp | Not in the future, not before 1990 |
| Derived fields | Candle size, body size, and wick calculations verified |
| Outliers | Statistical outlier detection (configurable threshold) |

---

## Logging

Logs are written to the `logs/` directory with automatic rotation:

| File | Description |
|---|---|
| `mt5_collector.log` | Main log — size-based rotation (10MB, 10 backups) |
| `mt5_daily.log` | Daily rotation for historical analysis |
| `error.log` | Errors only, for quick debugging |

### Log Levels

- **DEBUG** — Chunk processing details, validation warnings
- **INFO** — Collection progress, timeframe summaries
- **WARNING** — Invalid records, large candles, outliers
- **ERROR** — Connection failures, critical errors

---

## Automation

To schedule automatic data collection using Windows Task Scheduler:

1. Open **Task Scheduler**
2. Click **Create Basic Task**
3. Set your trigger (e.g. every hour)
4. Set the action to **Start a program**:
   - **Program:** `python`
   - **Arguments:** `C:\path\to\mt5-data-collector\mt5_collector.py`
   - **Start in:** `C:\path\to\mt5-data-collector`

---

## Troubleshooting

### Symbol not found

If you get `"Symbol 'USTech' not found"`, check available symbols on your broker:

```python
import MetaTrader5 as mt5

mt5.initialize()
symbols = mt5.symbols_get()
for s in symbols:
    if 'tech' in s.name.lower() or 'nas' in s.name.lower():
        print(s.name)
mt5.shutdown()
```

Common alternatives: `US100`, `USTEC`, `NAS100`, `NASDAQ100`

### MT5 not initializing

- Confirm MetaTrader 5 is installed and currently running
- Ensure you are logged into an account
- Verify the account has access to the target symbol

### PostgreSQL connection failed

- Confirm the PostgreSQL service is running
- Double-check credentials in your `.env` file
- Ensure the database user has `CREATE DATABASE` permissions

---

## File Structure

```
mt5-data-collector/
├── .env.example        # Environment variable template
├── .env                # Your configuration (create this)
├── requirements.txt    # Python dependencies
├── config.py           # Configuration loader
├── database.py         # Database setup and operations
├── mt5_collector.py    # Main data collector
├── query_data.py       # Query and export utilities
└── README.md           # This file
```

---

## License

MIT License — free for personal and commercial use.