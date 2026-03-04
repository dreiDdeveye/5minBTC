"""Polls Binance Futures top-trader long/short account ratio."""
import asyncio
import logging

import httpx

import config

logger = logging.getLogger(__name__)

ACCOUNT_RATIO_URL = f"{config.BINANCE_FUTURES_REST_BASE}/futures/data/topLongShortAccountRatio"


class LongShortPoller:
    def __init__(self):
        self.account_long_ratio: float = 0.5

    async def poll_loop(self, interval: int = 300):
        """Poll every `interval` seconds (data updates every 5m on Binance)."""
        while True:
            try:
                await self._fetch()
            except Exception as e:
                logger.warning(f"Long/short ratio fetch failed: {e}")
            await asyncio.sleep(interval)

    async def _fetch(self):
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(ACCOUNT_RATIO_URL, params={
                "symbol": "BTCUSDT", "period": "5m", "limit": 1,
            })
            resp.raise_for_status()
            data = resp.json()
            if data:
                self.account_long_ratio = float(data[-1].get("longAccount", 0.5))

    def get_snapshot(self) -> dict:
        return {
            "long_short_account_ratio": self.account_long_ratio,
        }
