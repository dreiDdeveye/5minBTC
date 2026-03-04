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


def atr_pct(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    """ATR as percentage of current price (normalized volatility)."""
    atr_val = atr(highs, lows, closes, period)
    if closes[-1] == 0:
        return 0.0
    return (atr_val / closes[-1]) * 100


def bollinger_width(closes: list[float], period: int = 20, num_std: float = 2.0) -> float:
    """(Upper Band - Lower Band) / Middle Band."""
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mean = sum(window) / period
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in window) / (period - 1)
    std = math.sqrt(variance)
    upper = mean + num_std * std
    lower = mean - num_std * std
    return (upper - lower) / mean


def bollinger_pos(closes: list[float], period: int = 20, num_std: float = 2.0) -> float:
    """Position within Bollinger Bands: (close - lower) / (upper - lower). 0=at lower, 1=at upper."""
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((x - mean) ** 2 for x in window) / (period - 1)
    std = math.sqrt(variance)
    upper = mean + num_std * std
    lower = mean - num_std * std
    band_width = upper - lower
    if band_width == 0:
        return 0.5
    return (closes[-1] - lower) / band_width
