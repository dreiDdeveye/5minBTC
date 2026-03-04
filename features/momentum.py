def return_nm(closes: list[float], n: int) -> float:
    """Percentage return over last n candles. closes[-1] is most recent."""
    if len(closes) < n + 1:
        return 0.0
    current = closes[-1]
    past = closes[-(n + 1)]
    if past == 0:
        return 0.0
    return (current - past) / past


def price_acceleration(closes: list[float]) -> float:
    """2nd derivative of price: change in 1m return. Positive = accelerating up."""
    if len(closes) < 3:
        return 0.0
    r1 = return_nm(closes[:-1], 1)  # previous 1m return
    r0 = return_nm(closes, 1)       # current 1m return
    return r0 - r1
