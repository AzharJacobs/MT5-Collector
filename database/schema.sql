-- MT5 OHLCV — Table Definitions
-- Reference copy of the live schema.
-- Authoritative migration script: storage/migrations/v1_initial.sql

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

CREATE TABLE IF NOT EXISTS ustech_features (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(18, 6), high DECIMAL(18, 6), low DECIMAL(18, 6),
    close DECIMAL(18, 6), volume DECIMAL(18, 6),
    demand_zone_top DECIMAL(18, 6), demand_zone_bottom DECIMAL(18, 6),
    demand_zone_strength DECIMAL(8, 4), demand_zone_fresh INTEGER,
    demand_zone_touches INTEGER, supply_zone_top DECIMAL(18, 6),
    supply_zone_bottom DECIMAL(18, 6), supply_zone_strength DECIMAL(8, 4),
    supply_zone_fresh INTEGER, supply_zone_touches INTEGER,
    nearest_demand_dist_atr DECIMAL(10, 4), nearest_supply_dist_atr DECIMAL(10, 4),
    in_demand_zone INTEGER, in_supply_zone INTEGER, between_zones INTEGER,
    bullish_engulfing INTEGER, bearish_engulfing INTEGER,
    pin_bar_bullish INTEGER, pin_bar_bearish INTEGER,
    higher_low INTEGER, lower_high INTEGER, bos_bullish INTEGER, bos_bearish INTEGER,
    buy_confirmation_score INTEGER, sell_confirmation_score INTEGER,
    atr_14 DECIMAL(18, 6), rsi_14 DECIMAL(8, 4),
    ema_20 DECIMAL(18, 6), ema_50 DECIMAL(18, 6), ema_200 DECIMAL(18, 6),
    ema_spread_atr DECIMAL(10, 4), price_above_ema20 INTEGER,
    price_above_ema50 INTEGER, price_above_ema200 INTEGER, ema_trend_bias INTEGER,
    bb_position DECIMAL(10, 4), bb_width_atr DECIMAL(10, 4),
    volume_ratio DECIMAL(10, 4), body_atr_ratio DECIMAL(10, 4),
    momentum_5 DECIMAL(10, 4), momentum_10 DECIMAL(10, 4),
    htf_1h_bias INTEGER, htf_4h_bias INTEGER, htf_aligned INTEGER,
    hour INTEGER, day_of_week TEXT, month INTEGER, session TEXT, direction TEXT,
    candle_size DECIMAL(18, 6), body_size DECIMAL(18, 6),
    wick_upper DECIMAL(18, 6), wick_lower DECIMAL(18, 6),
    signal INTEGER, signal_reason TEXT, trade_outcome INTEGER,
    label INTEGER, tp_price DECIMAL(18, 6), sl_price DECIMAL(18, 6), rr_ratio DECIMAL(8, 4),
    CONSTRAINT unique_features_symbol_tf_ts UNIQUE (symbol, timeframe, timestamp)
);
