import asyncio
import logging
from datetime import datetime, timezone, timedelta

from db import queries
from ingestion.ws_kline import KlineConsumer
from ingestion.ws_depth import DepthConsumer
from ingestion.ws_aggtrade import AggTradeConsumer
from ingestion import ws_ticker
from ingestion.rest_funding import FundingPoller
from ingestion.rest_openinterest import OpenInterestPoller
from ingestion.rest_longshort import LongShortPoller
from features.engine import compute_and_store_features
from features.target import backfill_targets
from model.predict import predict_latest

logger = logging.getLogger(__name__)

_candle_count = 0


async def run():
    global _candle_count
    depth_consumer = DepthConsumer()
    aggtrade_consumer = AggTradeConsumer()
    funding_poller = FundingPoller()
    oi_poller = OpenInterestPoller()
    ls_poller = LongShortPoller()

    async def handle_closed_candle(candle: dict):
        global _candle_count

        # Flush depth data for this candle
        depth_agg = depth_consumer.flush_and_reset()
        depth_agg["snapshot_time"] = candle["close_time"]

        # Flush aggregated trade data for this candle
        aggtrade_agg = aggtrade_consumer.flush_and_reset()
        aggtrade_agg["snapshot_time"] = candle["close_time"]

        # Store candle, depth, and aggtrade snapshots
        queries.upsert_kline(candle)
        queries.upsert_depth_snapshot(depth_agg)
        queries.insert_aggtrade_snapshot(aggtrade_agg)

        # Store futures snapshot (combines all REST-polled data)
        futures_snap = {
            "snapshot_time": candle["close_time"],
            **funding_poller.get_snapshot(),
            **oi_poller.get_snapshot(),
            **ls_poller.get_snapshot(),
        }
        queries.insert_futures_snapshot(futures_snap)

        _candle_count += 1
        logger.info(f"Candle #{_candle_count} stored. Close={candle['close']}")

        # Backfill targets in background (don't block the event loop)
        asyncio.create_task(backfill_targets())

        # Every 5 candles, compute features and predict
        if _candle_count % 5 == 0:
            logger.info("Computing features (5-candle block)...")
            feature_row = await compute_and_store_features(
                futures_data=[futures_snap],
                aggtrade_data=[aggtrade_agg],
            )
            if feature_row:
                result = await predict_latest(feature_row)
                if result:
                    logger.info(
                        f"Prediction: P={result['probability']:.3f} "
                        f"Signal={'LONG' if result['signal'] == 1 else 'HOLD'}"
                    )

    kline_consumer = KlineConsumer(on_candle_closed=handle_closed_candle)

    logger.info("Starting data collection (real-time price + 1m candles + depth + futures)...")
    await asyncio.gather(
        ws_ticker.connect(),
        kline_consumer.connect(),
        depth_consumer.connect(),
        aggtrade_consumer.connect(),
        funding_poller.poll_loop(60),
        oi_poller.poll_loop(60),
        ls_poller.poll_loop(300),
    )
