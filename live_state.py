"""Shared real-time state accessible by both ingestion and dashboard."""
import asyncio
import time

price: float = 0.0
high_24h: float = 0.0
low_24h: float = 0.0
volume_24h: float = 0.0
price_change_pct: float = 0.0
last_update: float = 0.0

# Subscribers for real-time push
_subscribers: set[asyncio.Queue] = set()


def update(p: float, h24: float = 0, l24: float = 0, v24: float = 0, chg: float = 0):
    global price, high_24h, low_24h, volume_24h, price_change_pct, last_update
    price = p
    if h24: high_24h = h24
    if l24: low_24h = l24
    if v24: volume_24h = v24
    price_change_pct = chg
    last_update = time.time()

    # Push to all subscribers
    snapshot = to_dict()
    dead = []
    for q in _subscribers:
        try:
            q.put_nowait(snapshot)
        except asyncio.QueueFull:
            pass
        except Exception:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)


def subscribe() -> asyncio.Queue:
    q = asyncio.Queue(maxsize=5)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue):
    _subscribers.discard(q)


polymarket: dict = {
    "up_price": 0.5,
    "down_price": 0.5,
    "spread": 0.0,
    "best_bid": 0.0,
    "best_ask": 0.0,
    "volume": 0.0,
    "liquidity": 0.0,
    "window_ts": 0,
}


def update_polymarket(snapshot: dict):
    global polymarket
    polymarket = snapshot


def to_dict() -> dict:
    return {
        "price": price,
        "high_24h": high_24h,
        "low_24h": low_24h,
        "volume_24h": volume_24h,
        "price_change_pct": price_change_pct,
        "last_update": last_update,
    }
