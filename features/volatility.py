import math


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    """Average True Range over `period` candles."""
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        trs.append(true_range(highs[i], lows[i], closes[i - 1]))

    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0
    return sum(trs[-period:]) / period


def bollinger_width(closes: list[float], period: int = 20, num_std: float = 2.0) -> float:
    """(Upper Band - Lower Band) / Middle Band."""
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mean = sum(window) / period
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = mean + num_std * std
    lower = mean - num_std * std
    return (upper - lower) / mean
