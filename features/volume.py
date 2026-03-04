def ratio(volumes: list[float], window: int) -> float:
    """Current volume / average of previous `window` candles."""
    if len(volumes) < window + 1:
        return 1.0
    current = volumes[-1]
    past = volumes[-(window + 1):-1]
    avg = sum(past) / len(past) if past else 1.0
    return current / avg if avg > 0 else 0.0


def spike_flag(volumes: list[float], window: int, threshold: float = 2.0) -> bool:
    return ratio(volumes, window) > threshold


def taker_buy_ratio(candles: list[dict], window: int = 5) -> float:
    """Ratio of taker buy volume to total volume over last `window` candles.
    > 0.5 = net buying pressure, < 0.5 = net selling pressure."""
    if not candles:
        return 0.5
    recent = candles[-window:]
    total_vol = sum(c.get("volume", 0) for c in recent)
    total_taker_buy = sum(c.get("taker_buy_vol", 0) for c in recent)
    if total_vol == 0:
        return 0.5
    return total_taker_buy / total_vol
