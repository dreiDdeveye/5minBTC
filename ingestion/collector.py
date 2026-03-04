import asyncio
import logging
from datetime import datetime, timezone, timedelta

import config
from db import queries
from ingestion.ws_kline import KlineConsumer
from ingestion.ws_depth import DepthConsumer
from ingestion.ws_aggtrade import AggTradeConsumer
from ingestion import ws_ticker
from ingestion.rest_funding import FundingPoller
from ingestion.rest_openinterest import OpenInterestPoller
from ingestion.rest_longshort import LongShortPoller
from ingestion.polymarket import PolymarketPoller
import live_state
from features.engine import compute_and_store_features
from features.target import backfill_targets
from model.predict import predict_latest, load_latest_model

logger = logging.getLogger(__name__)

_candle_count = 0
_features_since_retrain = 0
_retraining = False


def _is_5min_boundary(candle: dict) -> bool:
    """Check if this candle's close aligns with a UTC 5-minute boundary.
    A candle closing at :05, :10, :15, etc. marks the end of a 5-min block."""
    close_time = candle["close_time"]
    dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
    return dt.minute % 5 == 0


async def _maybe_retrain():
    """Retrain model if enough new data has accumulated."""
    global _features_since_retrain, _retraining
    if _retraining or _features_since_retrain < config.AUTO_RETRAIN_THRESHOLD:
        return

    _retraining = True
    logger.info(f"Auto-retraining triggered ({_features_since_retrain} new feature rows)...")

    try:
        from model.train import walk_forward_train
        model, fold_metrics, version = await asyncio.to_thread(walk_forward_train)
        if version:
            _features_since_retrain = 0
            # Reload the model in predict module
            load_latest_model()
            logger.info(f"Auto-retrain complete: {version}")
            if fold_metrics:
                avg_acc = sum(m["accuracy"] for m in fold_metrics) / len(fold_metrics)
                avg_f1 = sum(m["f1"] for m in fold_metrics) / len(fold_metrics)
                logger.info(f"Retrain metrics: avg_accuracy={avg_acc:.3f} avg_f1={avg_f1:.3f}")
        else:
            logger.warning("Auto-retrain produced no model")
    except Exception as e:
        logger.error(f"Auto-retrain failed: {e}")
    finally:
        _retraining = False


async def run():
    global _candle_count, _features_since_retrain
    depth_consumer = DepthConsumer()
    aggtrade_consumer = AggTradeConsumer()
    funding_poller = FundingPoller()
    oi_poller = OpenInterestPoller()
    ls_poller = LongShortPoller()
    polymarket_poller = PolymarketPoller()

    async def handle_closed_candle(candle: dict):
        global _candle_count, _features_since_retrain

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

        # Compute features on UTC 5-minute boundaries (:00, :05, :10, etc.)
        if _is_5min_boundary(candle):
            logger.info("Computing features (UTC 5-min boundary)...")
            feature_row = await compute_and_store_features(
                futures_data=[futures_snap],
                aggtrade_data=[aggtrade_agg],
                polymarket_data=polymarket_poller.get_snapshot(),
            )
            if feature_row:
                _features_since_retrain += 1
                result = await predict_latest(feature_row)
                if result:
                    sig_name = {1: 'OVER', -1: 'UNDER'}.get(result['signal'], '?')
                    logger.info(
                        f"Prediction: P={result['probability']:.3f} "
                        f"Signal={sig_name}"
                    )
                # Check if auto-retrain is needed
                asyncio.create_task(_maybe_retrain())

    kline_consumer = KlineConsumer(on_candle_closed=handle_closed_candle)

    async def polymarket_loop():
        """Poll Polymarket and push snapshots to live_state."""
        while True:
            try:
                await polymarket_poller._sync_market()
                if polymarket_poller.up_token_id:
                    await polymarket_poller._fetch_prices()
                live_state.update_polymarket(polymarket_poller.get_snapshot())
            except Exception as e:
                logger.warning(f"Polymarket poll failed: {e}")
            await asyncio.sleep(15)

    logger.info("Starting data collection (real-time price + 1m candles + depth + futures + polymarket)...")
    await asyncio.gather(
        ws_ticker.connect(),
        kline_consumer.connect(),
        depth_consumer.connect(),
        aggtrade_consumer.connect(),
        funding_poller.poll_loop(60),
        oi_poller.poll_loop(60),
        ls_poller.poll_loop(300),
        polymarket_loop(),
    )
