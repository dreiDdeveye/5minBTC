"""Polls Binance Futures premiumIndex for funding rate, mark price, index price."""
import asyncio
import logging

import httpx

import config

logger = logging.getLogger(__name__)

PREMIUM_INDEX_URL = f"{config.BINANCE_FUTURES_REST_BASE}/fapi/v1/premiumIndex"


class FundingPoller:
    def __init__(self):
        self.latest_funding_rate: float = 0.0
        self.latest_mark_price: float = 0.0
        self.latest_index_price: float = 0.0

    async def poll_loop(self, interval: int = 60):
        """Poll every `interval` seconds."""
        while True:
            try:
                await self._fetch()
            except Exception as e:
                logger.warning(f"Funding rate fetch failed: {e}")
            await asyncio.sleep(interval)

    async def _fetch(self):
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(PREMIUM_INDEX_URL, params={"symbol": "BTCUSDT"})
            resp.raise_for_status()
            data = resp.json()
            self.latest_funding_rate = float(data.get("lastFundingRate", 0))
            self.latest_mark_price = float(data.get("markPrice", 0))
            self.latest_index_price = float(data.get("indexPrice", 0))

    def get_snapshot(self) -> dict:
        return {
            "funding_rate": self.latest_funding_rate,
            "mark_price": self.latest_mark_price,
            "index_price": self.latest_index_price,
        }
