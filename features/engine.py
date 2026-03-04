import logging

from db import queries
from features import momentum, volume, volatility, trend, orderflow
from features import futures as futures_feat
from features import whale as whale_feat

logger = logging.getLogger(__name__)

# Need at least 30 candles for EMA(21) warmup + feature window
MIN_CANDLES = 30


async def compute_and_store_features(
    futures_data: list[dict] | None = None,
    aggtrade_data: list[dict] | None = None,
    polymarket_data: dict | None = None,
) -> dict | None:
    """Compute features from latest candles and store to Supabase. Returns the feature row."""
    candles = queries.get_latest_klines(MIN_CANDLES)
    if len(candles) < MIN_CANDLES:
        logger.warning(f"Not enough candles ({len(candles)}/{MIN_CANDLES}). Skipping.")
        return None

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    latest = candles[-1]

    # Get depth snapshots for the last 5 minutes using close_time range
    if len(candles) >= 5:
        depth_data = queries.get_depth_for_range(
            candles[-5]["close_time"], latest["close_time"]
        )
    else:
        depth_data = []

    # Use the 5-min window's opening candle open price as PTB baseline
    window_open = candles[-5]["open"] if len(candles) >= 5 else latest["open"]

    # Polymarket data (defaults to neutral)
    poly = polymarket_data or {}
    poly_up = poly.get("up_price", 0.5)
    poly_spread = poly.get("spread", 0.0)

    feature_row = {
        "feature_time": latest["open_time"],
        "close_price": latest["close"],
        "open_price": window_open,
        "return_1m": momentum.return_nm(closes, 1),
        "return_3m": momentum.return_nm(closes, 3),
        "return_5m": momentum.return_nm(closes, 5),
        "volume_ratio_20": volume.ratio(volumes, 20),
        "volume_spike": volume.spike_flag(volumes, 20),
        "atr_7": volatility.atr(highs, lows, closes, 7),
        "atr_pct": volatility.atr_pct(highs, lows, closes, 7),
        "bollinger_width": volatility.bollinger_width(closes, 20),
        "bollinger_pos": volatility.bollinger_pos(closes, 20),
        "ema_cross": trend.ema_cross(closes, 9, 21),
        "rsi_7": trend.rsi(closes, 7),
        "order_flow_imbalance": orderflow.imbalance(depth_data),
        "price_acceleration": momentum.price_acceleration(closes),
        # Binance Futures features
        "funding_rate": futures_feat.funding_rate_feature(futures_data or []),
        "oi_change_pct": futures_feat.oi_change_feature(futures_data or []),
        "long_short_ratio": futures_feat.long_short_ratio_feature(futures_data or []),
        "taker_buy_ratio": volume.taker_buy_ratio(candles, 5),
        "whale_ratio": whale_feat.whale_ratio_feature(aggtrade_data or []),
        "whale_net_flow": whale_feat.whale_net_flow_feature(aggtrade_data or []),
        # Polymarket features
        "poly_up_price": poly_up,
        "poly_spread": poly_spread,
        "target": None,
    }

    queries.insert_feature_row(feature_row)
    logger.info(f"Features stored for {latest['close_time']}")
    return feature_row
