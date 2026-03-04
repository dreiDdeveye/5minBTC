import asyncio
import logging
from datetime import datetime, timedelta, timezone

from db import queries

logger = logging.getLogger(__name__)


def _backfill_sync():
    """Synchronous backfill - runs in a thread to avoid blocking the event loop."""
    rows = queries.get_features_without_target(limit=20)
    filled = 0
    for row in rows:
        ft = row["feature_time"]
        # Skip old stale rows with close_time format (contain .999)
        if ".999" in ft:
            continue

        target_time = (
            datetime.fromisoformat(ft.replace("Z", "+00:00")) + timedelta(minutes=5)
        ).isoformat()

        future_candle = queries.get_kline_at_time(target_time)
        if future_candle is None:
            continue  # outcome hasn't happened yet

        target = 1 if future_candle["close"] > row["close_price"] else 0
        queries.backfill_target(ft, target)
        filled += 1
        logger.debug(f"Backfilled target={target} for feature_time={ft}")

    if filled:
        logger.info(f"Backfilled {filled} feature targets")


def _backfill_predictions_sync():
    """Backfill actual_target on predictions that are 5+ minutes old."""
    preds = queries.get_predictions_without_outcome(limit=20)
    filled = 0
    for pred in preds:
        ft = pred["feature_time"]
        target_time = (
            datetime.fromisoformat(ft.replace("Z", "+00:00")) + timedelta(minutes=5)
        ).isoformat()

        future_candle = queries.get_kline_at_time(target_time)
        if future_candle is None:
            continue

        # Get the close price at prediction time
        entry_candle = queries.get_kline_at_time(ft)
        if entry_candle is None:
            continue

        actual = 1 if future_candle["close"] > entry_candle["close"] else 0
        queries.update_prediction_outcome(pred["id"], actual)
        filled += 1
        logger.debug(f"Prediction #{pred['id']}: actual={actual}")

    if filled:
        logger.info(f"Backfilled {filled} prediction outcomes")


async def backfill_targets():
    """Backfill both feature targets and prediction outcomes.
    Runs in a thread pool to avoid blocking the async event loop."""
    await asyncio.to_thread(_backfill_sync)
    await asyncio.to_thread(_backfill_predictions_sync)
