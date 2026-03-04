"""
Fetch historical 1m BTC candles from Binance REST API,
compute features, backfill targets, and train the initial model.

Usage:
    python -m scripts.seed_historical --days 14
"""
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

import config
from db import client, queries
from features import momentum, volume, volatility, trend, orderflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KLINES_URL = f"{config.BINANCE_REST_BASE}/api/v3/klines"
FUNDING_HIST_URL = f"{config.BINANCE_FUTURES_REST_BASE}/fapi/v1/fundingRate"
OI_HIST_URL = f"{config.BINANCE_FUTURES_REST_BASE}/futures/data/openInterestHist"
LS_HIST_URL = f"{config.BINANCE_FUTURES_REST_BASE}/futures/data/topLongShortAccountRatio"
BATCH_SIZE = 1000  # Binance max per request


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch klines from Binance REST API."""
    all_klines = []
    current_start = start_ms

    with httpx.Client(timeout=30) as client:
        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": BATCH_SIZE,
            }
            resp = client.get(KLINES_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_klines.extend(data)
            # Move start to after the last candle
            current_start = data[-1][0] + 60000  # +1 minute in ms
            logger.info(f"Fetched {len(all_klines)} candles so far...")
            time.sleep(0.2)  # rate limit

    return all_klines


def fetch_historical_funding(start_ms: int, end_ms: int) -> dict[str, float]:
    """Fetch historical funding rates. Returns {timestamp_iso: rate}."""
    result = {}
    current_start = start_ms
    with httpx.Client(timeout=30) as http:
        while current_start < end_ms:
            try:
                resp = http.get(FUNDING_HIST_URL, params={
                    "symbol": "BTCUSDT",
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 1000,
                })
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                for row in data:
                    ts = datetime.fromtimestamp(row["fundingTime"] / 1000, tz=timezone.utc).isoformat()
                    result[ts] = float(row["fundingRate"])
                current_start = data[-1]["fundingTime"] + 1
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Funding rate fetch failed: {e}")
                break
    logger.info(f"Fetched {len(result)} historical funding rates")
    return result


def fetch_historical_oi(start_ms: int, end_ms: int) -> list[dict]:
    """Fetch historical open interest (5m periods). Returns list of {ts_ms, oi}."""
    result = []
    current_start = start_ms
    with httpx.Client(timeout=30) as http:
        while current_start < end_ms:
            try:
                resp = http.get(OI_HIST_URL, params={
                    "symbol": "BTCUSDT",
                    "period": "5m",
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 500,
                })
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                for row in data:
                    result.append({
                        "ts_ms": row["timestamp"],
                        "oi": float(row["sumOpenInterest"]),
                    })
                current_start = data[-1]["timestamp"] + 1
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"OI history fetch failed: {e}")
                break
    logger.info(f"Fetched {len(result)} historical OI data points")
    return result


def fetch_historical_longshort(start_ms: int, end_ms: int) -> list[dict]:
    """Fetch historical long/short ratios (5m periods)."""
    result = []
    current_start = start_ms
    with httpx.Client(timeout=30) as http:
        while current_start < end_ms:
            try:
                resp = http.get(LS_HIST_URL, params={
                    "symbol": "BTCUSDT",
                    "period": "5m",
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 500,
                })
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                for row in data:
                    result.append({
                        "ts_ms": row["timestamp"],
                        "long_ratio": float(row["longAccount"]),
                    })
                current_start = data[-1]["timestamp"] + 1
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Long/short history fetch failed: {e}")
                break
    logger.info(f"Fetched {len(result)} historical long/short data points")
    return result


def build_futures_lookup(
    funding: dict[str, float],
    oi_data: list[dict],
    ls_data: list[dict],
) -> dict:
    """Build a lookup: feature_time_ms -> {funding_rate, oi_change_pct, long_short_ratio}."""
    # OI: compute change pct relative to previous reading
    oi_by_ms = {}
    for i, row in enumerate(oi_data):
        if i == 0:
            oi_by_ms[row["ts_ms"]] = 0.0
        else:
            prev = oi_data[i - 1]["oi"]
            oi_by_ms[row["ts_ms"]] = (row["oi"] - prev) / prev if prev > 0 else 0.0

    # Long/short by timestamp
    ls_by_ms = {row["ts_ms"]: row["long_ratio"] for row in ls_data}

    # Funding: keyed by ISO timestamp, convert to ms for lookup
    funding_by_ms = {}
    for ts_iso, rate in funding.items():
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        funding_by_ms[int(dt.timestamp() * 1000)] = rate

    return {
        "funding_by_ms": funding_by_ms,
        "oi_by_ms": oi_by_ms,
        "ls_by_ms": ls_by_ms,
    }


def lookup_futures_at_time(lookup: dict, feature_time_iso: str) -> dict:
    """Find the closest futures data at or before feature_time."""
    dt = datetime.fromisoformat(feature_time_iso.replace("Z", "+00:00"))
    ts_ms = int(dt.timestamp() * 1000)

    # Funding: find most recent rate at or before this time
    funding_rate = 0.0
    funding_times = sorted(lookup["funding_by_ms"].keys())
    for t in reversed(funding_times):
        if t <= ts_ms:
            funding_rate = lookup["funding_by_ms"][t]
            break

    # OI change: find closest 5m bucket
    oi_change = 0.0
    oi_times = sorted(lookup["oi_by_ms"].keys())
    for t in reversed(oi_times):
        if t <= ts_ms:
            oi_change = lookup["oi_by_ms"][t]
            break

    # Long/short ratio: find closest 5m bucket
    ls_ratio = 0.5
    ls_times = sorted(lookup["ls_by_ms"].keys())
    for t in reversed(ls_times):
        if t <= ts_ms:
            ls_ratio = lookup["ls_by_ms"][t]
            break

    return {
        "funding_rate": funding_rate,
        "oi_change_pct": oi_change,
        "long_short_ratio": ls_ratio,
    }


def parse_kline(raw: list) -> dict:
    return {
        "open_time": datetime.fromtimestamp(raw[0] / 1000, tz=timezone.utc).isoformat(),
        "close_time": datetime.fromtimestamp(raw[6] / 1000, tz=timezone.utc).isoformat(),
        "open": float(raw[1]),
        "high": float(raw[2]),
        "low": float(raw[3]),
        "close": float(raw[4]),
        "volume": float(raw[5]),
        "quote_volume": float(raw[7]),
        "trade_count": int(raw[8]),
        "taker_buy_vol": float(raw[9]),
        "taker_buy_quote": float(raw[10]),
    }


def compute_features_batch(candles: list[dict], futures_lookup: dict | None = None) -> list[dict]:
    """Compute features for every 5-candle block in the historical data."""
    features = []
    min_candles = 30  # need EMA(21) warmup

    for i in range(min_candles - 1, len(candles), 5):
        if i >= len(candles):
            break
        window = candles[max(0, i - min_candles + 1):i + 1]
        if len(window) < min_candles:
            continue

        closes = [c["close"] for c in window]
        highs = [c["high"] for c in window]
        lows = [c["low"] for c in window]
        volumes = [c["volume"] for c in window]
        latest = window[-1]

        # Futures data lookup
        if futures_lookup:
            fut = lookup_futures_at_time(futures_lookup, latest["open_time"])
        else:
            fut = {"funding_rate": 0.0, "oi_change_pct": 0.0, "long_short_ratio": 0.5}

        feature_row = {
            "feature_time": latest["open_time"],
            "close_price": latest["close"],
            "return_1m": momentum.return_nm(closes, 1),
            "return_3m": momentum.return_nm(closes, 3),
            "return_5m": momentum.return_nm(closes, 5),
            "volume_ratio_20": volume.ratio(volumes, 20),
            "volume_spike": volume.spike_flag(volumes, 20),
            "atr_7": volatility.atr(highs, lows, closes, 7),
            "bollinger_width": volatility.bollinger_width(closes, 20),
            "ema_cross": trend.ema_cross(closes, 9, 21),
            "rsi_7": trend.rsi(closes, 7),
            "order_flow_imbalance": 0.0,  # no depth data for historical
            # New Binance Futures features
            "funding_rate": fut["funding_rate"],
            "oi_change_pct": fut["oi_change_pct"],
            "long_short_ratio": fut["long_short_ratio"],
            "taker_buy_ratio": volume.taker_buy_ratio(window, 5),
            "whale_ratio": 0.0,      # no historical aggTrade data available
            "whale_net_flow": 0.0,    # no historical aggTrade data available
            "target": None,
        }
        features.append(feature_row)

    return features


def backfill_targets_batch(features: list[dict], candles: list[dict]):
    """Compute targets using the historical candle data."""
    # Build lookup: open_time -> candle
    candle_by_time = {}
    for c in candles:
        candle_by_time[c["open_time"]] = c

    for feat in features:
        ft = feat["feature_time"]
        target_time = (
            datetime.fromisoformat(ft.replace("Z", "+00:00")) + timedelta(minutes=5)
        ).isoformat()

        future = candle_by_time.get(target_time)
        if future:
            feat["target"] = 1 if future["close"] > feat["close_price"] else 0


def main(days: int = 14):
    logger.info(f"Seeding {days} days of historical 1m BTC candles...")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    # 1. Fetch candles
    raw_klines = fetch_klines("BTCUSDT", "1m", start_ms, end_ms)
    candles = [parse_kline(k) for k in raw_klines]
    logger.info(f"Total candles fetched: {len(candles)}")

    # 2. Store candles in Supabase (bulk upsert)
    logger.info("Storing candles in Supabase (bulk)...")
    batch_size = 500
    for i in range(0, len(candles), batch_size):
        batch = candles[i:i + batch_size]
        client.upsert("raw_klines", batch, on_conflict="open_time")
        logger.info(f"Stored {min(i + batch_size, len(candles))}/{len(candles)} candles")

    # 3. Fetch historical futures data
    logger.info("Fetching historical futures data (funding, OI, long/short)...")
    funding = fetch_historical_funding(start_ms, end_ms)
    oi_data = fetch_historical_oi(start_ms, end_ms)
    ls_data = fetch_historical_longshort(start_ms, end_ms)
    futures_lookup = build_futures_lookup(funding, oi_data, ls_data)

    # 4. Compute features
    logger.info("Computing features...")
    features = compute_features_batch(candles, futures_lookup)
    logger.info(f"Computed {len(features)} feature rows")

    # 5. Backfill targets
    logger.info("Backfilling targets...")
    backfill_targets_batch(features, candles)
    with_target = [f for f in features if f["target"] is not None]
    logger.info(f"Features with target: {len(with_target)}/{len(features)}")

    # 6. Store features (bulk)
    logger.info("Storing features in Supabase (bulk)...")
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        client.upsert("features", batch, on_conflict="feature_time")
        logger.info(f"Stored {min(i + batch_size, len(features))}/{len(features)} features")

    # 7. Train model
    logger.info("Training initial model...")
    from model.train import walk_forward_train
    model, fold_metrics, version = walk_forward_train()
    if version:
        logger.info(f"Model trained: {version}")
        if fold_metrics:
            avg_acc = sum(m["accuracy"] for m in fold_metrics) / len(fold_metrics)
            logger.info(f"Average accuracy across {len(fold_metrics)} folds: {avg_acc:.3f}")
    else:
        logger.warning("No model was trained (not enough data)")

    logger.info("Seeding complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="Days of history to fetch")
    args = parser.parse_args()
    main(args.days)
