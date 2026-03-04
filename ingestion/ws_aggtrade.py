"""Aggregated trade stream – tracks taker buy/sell volumes and whale activity."""
import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

import config

logger = logging.getLogger(__name__)


class AggTradeConsumer:
    def __init__(self):
        self.url = f"{config.BINANCE_WS_BASE}/{config.AGGTRADE_STREAM}"
        self._total_buy_vol = 0.0
        self._total_sell_vol = 0.0
        self._whale_buy_vol = 0.0
        self._whale_sell_vol = 0.0
        self._whale_count = 0
        self._trade_count = 0

    async def connect(self):
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    logger.info("AggTrade WebSocket connected")
                    backoff = 1
                    async for raw in ws:
                        self._accumulate(json.loads(raw))
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"AggTrade WS disconnected: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _accumulate(self, msg: dict):
        qty = float(msg.get("q", 0))
        is_buyer_maker = msg.get("m", False)  # True = sell (maker is buyer), False = buy

        if is_buyer_maker:
            self._total_sell_vol += qty
        else:
            self._total_buy_vol += qty

        if qty >= config.WHALE_THRESHOLD_BTC:
            self._whale_count += 1
            if is_buyer_maker:
                self._whale_sell_vol += qty
            else:
                self._whale_buy_vol += qty

        self._trade_count += 1

    def flush_and_reset(self) -> dict:
        total = self._total_buy_vol + self._total_sell_vol
        whale_total = self._whale_buy_vol + self._whale_sell_vol

        result = {
            "snapshot_time": datetime.now(timezone.utc).isoformat(),
            "total_buy_vol": self._total_buy_vol,
            "total_sell_vol": self._total_sell_vol,
            "whale_buy_vol": self._whale_buy_vol,
            "whale_sell_vol": self._whale_sell_vol,
            "whale_count": self._whale_count,
            "trade_count": self._trade_count,
            "whale_ratio": whale_total / total if total > 0 else 0.0,
            "whale_net_flow": (
                (self._whale_buy_vol - self._whale_sell_vol) / whale_total
                if whale_total > 0 else 0.0
            ),
        }

        # Reset accumulators
        self._total_buy_vol = 0.0
        self._total_sell_vol = 0.0
        self._whale_buy_vol = 0.0
        self._whale_sell_vol = 0.0
        self._whale_count = 0
        self._trade_count = 0

        return result
