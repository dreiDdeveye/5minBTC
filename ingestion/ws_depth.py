import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

import config

logger = logging.getLogger(__name__)


class DepthConsumer:
    def __init__(self):
        self.url = f"{config.BINANCE_WS_BASE}/{config.DEPTH_STREAM}"
        self._bid_sum = 0.0
        self._ask_sum = 0.0
        self._best_bid_sum = 0.0
        self._best_ask_sum = 0.0
        self._spread_sum = 0.0
        self._tick_count = 0

    async def connect(self):
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    logger.info("Depth WebSocket connected")
                    backoff = 1
                    async for raw in ws:
                        self._accumulate(json.loads(raw))
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"Depth WS disconnected: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _accumulate(self, msg: dict):
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        if not bids or not asks:
            return

        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])

        self._bid_sum += bid_vol
        self._ask_sum += ask_vol
        self._best_bid_sum += best_bid
        self._best_ask_sum += best_ask
        self._spread_sum += best_ask - best_bid
        self._tick_count += 1

    def flush_and_reset(self) -> dict:
        if self._tick_count == 0:
            now = datetime.now(timezone.utc).isoformat()
            return {
                "snapshot_time": now,
                "best_bid": 0.0,
                "best_ask": 0.0,
                "bid_volume_total": 0.0,
                "ask_volume_total": 0.0,
                "spread": 0.0,
            }

        n = self._tick_count
        result = {
            "snapshot_time": datetime.now(timezone.utc).isoformat(),
            "best_bid": self._best_bid_sum / n,
            "best_ask": self._best_ask_sum / n,
            "bid_volume_total": self._bid_sum / n,
            "ask_volume_total": self._ask_sum / n,
            "spread": self._spread_sum / n,
        }

        self._bid_sum = 0.0
        self._ask_sum = 0.0
        self._best_bid_sum = 0.0
        self._best_ask_sum = 0.0
        self._spread_sum = 0.0
        self._tick_count = 0

        return result
