def ema(values: list[float], period: int) -> float:
    """Compute EMA and return the final value."""
    if not values:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)

    multiplier = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    ema_val = sum(values[:period]) / period
    for v in values[period:]:
        ema_val = (v - ema_val) * multiplier + ema_val
    return ema_val


def ema_cross(closes: list[float], fast: int = 9, slow: int = 21) -> float:
    """EMA(fast) - EMA(slow). Positive = bullish tendency."""
    return ema(closes, fast) - ema(closes, slow)


def rsi(closes: list[float], period: int = 7) -> float:
    """Relative Strength Index using Wilder smoothing."""
    if len(closes) < period + 1:
        return 50.0  # neutral default

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed with SMA of first `period` deltas
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for remaining deltas
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(d, 0))) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
