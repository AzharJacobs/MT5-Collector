-- MT5 OHLCV Data Collector - Database Schema
-- Run this manually if you prefer direct SQL setup

-- Create database (run this first, separately)
-- CREATE DATABASE ustech_data;

-- Connect to the database first:
-- \c ustech_data

-- Create main table
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

-- Create index for fast queries on timeframe and timestamp
CREATE INDEX IF NOT EXISTS idx_timeframe_timestamp
ON ustech_ohlcv (timeframe, timestamp DESC);

-- Create view with proper timeframe ordering
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
    END,
    timestamp DESC;

-- Useful queries for verification:

-- Check total rows per timeframe
-- SELECT timeframe, COUNT(*) as total, MIN(timestamp) as earliest, MAX(timestamp) as latest
-- FROM ustech_ohlcv
-- GROUP BY timeframe
-- ORDER BY
--     CASE timeframe
--         WHEN '1min' THEN 1 WHEN '2min' THEN 2 WHEN '3min' THEN 3
--         WHEN '4min' THEN 4 WHEN '5min' THEN 5 WHEN '10min' THEN 6
--         WHEN '15min' THEN 7 WHEN '30min' THEN 8 WHEN '1H' THEN 9
--         WHEN '4H' THEN 10 WHEN '1D' THEN 11 ELSE 99
--     END;

-- Check direction distribution
-- SELECT timeframe, direction, COUNT(*) as count
-- FROM ustech_ohlcv
-- GROUP BY timeframe, direction
-- ORDER BY timeframe, direction;

-- Get latest candles from the view
-- SELECT * FROM ustech_view LIMIT 100;
