"""Pull live BTC 5-minute market data from Polymarket CLOB + Gamma APIs."""
import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import websockets

import config

logger = logging.getLogger(__name__)


def _current_window_ts() -> int:
    """Unix timestamp of the current UTC 5-minute window start."""
    now = int(datetime.now(timezone.utc).timestamp())
    return now - (now % 300)


class PolymarketPoller:
    """Fetches live Polymarket odds for the current BTC 5-min Up/Down market."""

    def __init__(self):
        self.up_token_id: str | None = None
        self.down_token_id: str | None = None
        self.condition_id: str | None = None
        self.current_window_ts: int = 0
        # Live market data
        self.up_price: float = 0.5
        self.down_price: float = 0.5
        self.spread: float = 0.0
        self.volume: float = 0.0
        self.liquidity: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0

    async def poll_loop(self, interval: int = 15):
        """Poll Polymarket every `interval` seconds."""
        while True:
            try:
                await self._sync_market()
                if self.up_token_id:
                    await self._fetch_prices()
            except Exception as e:
                logger.warning(f"Polymarket poll failed: {e}")
            await asyncio.sleep(interval)

    async def _sync_market(self):
        """Ensure we have the token IDs for the current 5-min window."""
        window_ts = _current_window_ts()
        if window_ts == self.current_window_ts and self.up_token_id:
            return  # already synced

        self.current_window_ts = window_ts
        slug = f"btc-updown-5m-{window_ts}"

        async with httpx.AsyncClient(timeout=10) as client:
            # Try Gamma API for event discovery
            resp = await client.get(
                f"{config.POLYMARKET_GAMMA_BASE}/events",
                params={"slug": slug, "active": "true", "closed": "false"},
            )
            resp.raise_for_status()
            events = resp.json()

            if not events:
                logger.debug(f"No Polymarket event found for {slug}")
                self.up_token_id = None
                self.down_token_id = None
                return

            event = events[0] if isinstance(events, list) else events
            markets = event.get("markets", [])
            if not markets:
                logger.debug(f"No markets in event {slug}")
                return

            market = markets[0]
            self.condition_id = market.get("conditionId")
            self.volume = float(market.get("volume", 0) or 0)
            self.liquidity = float(market.get("liquidity", 0) or 0)

            # Parse clobTokenIds — JSON string with [up_token, down_token]
            token_ids_raw = market.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
            except json.JSONDecodeError:
                token_ids = []

            if len(token_ids) >= 2:
                self.up_token_id = token_ids[0]
                self.down_token_id = token_ids[1]
                logger.info(f"Polymarket synced: {slug} (Up={self.up_token_id[:12]}...)")
            else:
                logger.warning(f"Unexpected token_ids format: {token_ids_raw}")

            # Also grab outcomePrices if available
            outcome_prices = market.get("outcomePrices")
            if outcome_prices:
                try:
                    prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    if len(prices) >= 2:
                        self.up_price = float(prices[0])
                        self.down_price = float(prices[1])
                except (json.JSONDecodeError, ValueError):
                    pass

    async def _fetch_prices(self):
        """Fetch live prices from CLOB API."""
        async with httpx.AsyncClient(timeout=10) as client:
            # Fetch midpoint prices for both tokens
            try:
                resp = await client.get(
                    f"{config.POLYMARKET_CLOB_BASE}/midpoint",
                    params={"token_id": self.up_token_id},
                )
                resp.raise_for_status()
                data = resp.json()
                self.up_price = float(data.get("mid", self.up_price))
            except Exception:
                pass

            try:
                resp = await client.get(
                    f"{config.POLYMARKET_CLOB_BASE}/midpoint",
                    params={"token_id": self.down_token_id},
                )
                resp.raise_for_status()
                data = resp.json()
                self.down_price = float(data.get("mid", self.down_price))
            except Exception:
                pass

            # Fetch spread for Up token
            try:
                resp = await client.get(
                    f"{config.POLYMARKET_CLOB_BASE}/spread",
                    params={"token_id": self.up_token_id},
                )
                resp.raise_for_status()
                data = resp.json()
                self.spread = float(data.get("spread", 0))
                self.best_bid = float(data.get("bid", 0) or 0)
                self.best_ask = float(data.get("ask", 0) or 0)
            except Exception:
                pass

    def get_snapshot(self) -> dict:
        return {
            "up_price": self.up_price,
            "down_price": self.down_price,
            "spread": self.spread,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "window_ts": self.current_window_ts,
        }
