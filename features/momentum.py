def return_nm(closes: list[float], n: int) -> float:
    """Percentage return over last n candles. closes[-1] is most recent."""
    if len(closes) < n + 1:
        return 0.0
    current = closes[-1]
    past = closes[-(n + 1)]
    if past == 0:
        return 0.0
    return (current - past) / past
