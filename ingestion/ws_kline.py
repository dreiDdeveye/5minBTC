import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

import config

logger = logging.getLogger(__name__)


class KlineConsumer:
    def __init__(self, on_candle_closed):
        self.on_candle_closed = on_candle_closed
        self.url = f"{config.BINANCE_WS_BASE}/{config.KLINE_STREAM}"
        self.current_price = None

    async def connect(self):
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    logger.info("Kline WebSocket connected")
                    backoff = 1
                    async for raw in ws:
                        await self._handle(json.loads(raw))
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"Kline WS disconnected: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _handle(self, msg: dict):
        k = msg.get("k", {})
        self.current_price = float(k.get("c", 0))

        if not k.get("x"):  # candle not closed yet
            return

        candle = {
            "open_time": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc).isoformat(),
            "close_time": datetime.fromtimestamp(k["T"] / 1000, tz=timezone.utc).isoformat(),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "quote_volume": float(k["q"]),
            "trade_count": int(k["n"]),
            "taker_buy_vol": float(k["V"]),
            "taker_buy_quote": float(k["Q"]),
        }
        logger.info(f"Candle closed: {candle['close_time']} close={candle['close']}")
        await self.on_candle_closed(candle)
