"""Real-time BTC price via Binance miniTicker stream (~1s updates)."""
import asyncio
import json
import logging

import websockets

import config
import live_state

logger = logging.getLogger(__name__)

TICKER_URL = f"{config.BINANCE_WS_BASE}/btcusdt@miniTicker"


async def connect():
    backoff = 1
    while True:
        try:
            async with websockets.connect(TICKER_URL) as ws:
                logger.info("Ticker WebSocket connected (real-time price)")
                backoff = 1
                async for raw in ws:
                    data = json.loads(raw)
                    live_state.update(
                        p=float(data["c"]),     # close / last price
                        h24=float(data["h"]),   # 24h high
                        l24=float(data["l"]),   # 24h low
                        v24=float(data["v"]),   # 24h volume
                        chg=_pct(data),
                    )
        except Exception as e:
            logger.warning(f"Ticker WS error: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def _pct(data: dict) -> float:
    o = float(data.get("o", 0))
    c = float(data.get("c", 0))
    if o == 0:
        return 0.0
    return ((c - o) / o) * 100
