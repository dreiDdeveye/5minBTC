"""Polls Binance Futures openInterest for BTC perpetual."""
import asyncio
import logging

import httpx

import config

logger = logging.getLogger(__name__)

OI_URL = f"{config.BINANCE_FUTURES_REST_BASE}/fapi/v1/openInterest"


class OpenInterestPoller:
    def __init__(self):
        self._oi_history: list[float] = []
        self.latest_oi: float = 0.0

    async def poll_loop(self, interval: int = 60):
        """Poll every `interval` seconds."""
        while True:
            try:
                await self._fetch()
            except Exception as e:
                logger.warning(f"Open interest fetch failed: {e}")
            await asyncio.sleep(interval)

    async def _fetch(self):
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(OI_URL, params={"symbol": "BTCUSDT"})
            resp.raise_for_status()
            data = resp.json()
            self.latest_oi = float(data.get("openInterest", 0))
            self._oi_history.append(self.latest_oi)
            if len(self._oi_history) > 10:
                self._oi_history = self._oi_history[-10:]

    def get_snapshot(self) -> dict:
        return {
            "open_interest": self.latest_oi,
            "oi_change_pct": self._oi_change_pct(),
        }

    def _oi_change_pct(self) -> float:
        if len(self._oi_history) < 2:
            return 0.0
        prev = self._oi_history[0]
        if prev == 0:
            return 0.0
        return (self.latest_oi - prev) / prev
