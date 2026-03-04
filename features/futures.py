"""Features derived from Binance Futures data (funding rate, OI, long/short)."""


def funding_rate_feature(futures_data: list[dict]) -> float:
    """Latest funding rate from futures snapshots."""
    if not futures_data:
        return 0.0
    return futures_data[-1].get("funding_rate", 0.0)


def oi_change_feature(futures_data: list[dict]) -> float:
    """Open interest change percentage from latest snapshot."""
    if not futures_data:
        return 0.0
    return futures_data[-1].get("oi_change_pct", 0.0)


def long_short_ratio_feature(futures_data: list[dict]) -> float:
    """Top trader long/short account ratio. 0.5 = neutral."""
    if not futures_data:
        return 0.5
    return futures_data[-1].get("long_short_account_ratio", 0.5)
