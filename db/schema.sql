-- Run this in your Supabase SQL Editor to create all tables

CREATE TABLE IF NOT EXISTS raw_klines (
    id              BIGSERIAL PRIMARY KEY,
    open_time       TIMESTAMPTZ NOT NULL,
    close_time      TIMESTAMPTZ NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    quote_volume    DOUBLE PRECISION NOT NULL,
    trade_count     INTEGER NOT NULL,
    taker_buy_vol   DOUBLE PRECISION NOT NULL,
    taker_buy_quote DOUBLE PRECISION NOT NULL,
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(open_time)
);
CREATE INDEX IF NOT EXISTS idx_klines_open_time ON raw_klines (open_time DESC);

CREATE TABLE IF NOT EXISTS depth_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    snapshot_time    TIMESTAMPTZ NOT NULL,
    best_bid         DOUBLE PRECISION NOT NULL,
    best_ask         DOUBLE PRECISION NOT NULL,
    bid_volume_total DOUBLE PRECISION NOT NULL,
    ask_volume_total DOUBLE PRECISION NOT NULL,
    spread           DOUBLE PRECISION NOT NULL,
    inserted_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_depth_time ON depth_snapshots (snapshot_time DESC);

CREATE TABLE IF NOT EXISTS features (
    id                   BIGSERIAL PRIMARY KEY,
    feature_time         TIMESTAMPTZ NOT NULL,
    close_price          DOUBLE PRECISION NOT NULL,
    return_1m            DOUBLE PRECISION,
    return_3m            DOUBLE PRECISION,
    return_5m            DOUBLE PRECISION,
    volume_ratio_20      DOUBLE PRECISION,
    volume_spike         BOOLEAN,
    atr_7                DOUBLE PRECISION,
    bollinger_width      DOUBLE PRECISION,
    ema_cross            DOUBLE PRECISION,
    rsi_7                DOUBLE PRECISION,
    order_flow_imbalance DOUBLE PRECISION,
    target               SMALLINT,
    inserted_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(feature_time)
);
CREATE INDEX IF NOT EXISTS idx_features_time ON features (feature_time DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    predicted_at    TIMESTAMPTZ NOT NULL,
    feature_time    TIMESTAMPTZ NOT NULL,
    model_version   TEXT NOT NULL,
    probability     DOUBLE PRECISION NOT NULL,
    signal          SMALLINT NOT NULL,
    actual_target   SMALLINT,
    inserted_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions (predicted_at DESC);

CREATE TABLE IF NOT EXISTS model_metrics (
    id              BIGSERIAL PRIMARY KEY,
    model_version   TEXT NOT NULL,
    fold_index      INTEGER,
    train_start     TIMESTAMPTZ,
    train_end       TIMESTAMPTZ,
    test_start      TIMESTAMPTZ,
    test_end        TIMESTAMPTZ,
    accuracy        DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall          DOUBLE PRECISION,
    f1              DOUBLE PRECISION,
    brier_score     DOUBLE PRECISION,
    roc_auc         DOUBLE PRECISION,
    n_samples_train INTEGER,
    n_samples_test  INTEGER,
    inserted_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id              BIGSERIAL PRIMARY KEY,
    backtest_run    TEXT NOT NULL,
    signal_time     TIMESTAMPTZ NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION NOT NULL,
    probability     DOUBLE PRECISION NOT NULL,
    signal          SMALLINT NOT NULL,
    actual_target   SMALLINT NOT NULL,
    correct         BOOLEAN NOT NULL,
    inserted_at     TIMESTAMPTZ DEFAULT NOW()
);
