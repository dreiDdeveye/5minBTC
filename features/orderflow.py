def imbalance(depth_snapshots: list[dict]) -> float:
    """Average order flow imbalance: (bid_vol - ask_vol) / total_vol."""
    if not depth_snapshots:
        return 0.0
    values = []
    for snap in depth_snapshots:
        bid = snap.get("bid_volume_total", 0)
        ask = snap.get("ask_volume_total", 0)
        total = bid + ask
        if total > 0:
            values.append((bid - ask) / total)
    return sum(values) / len(values) if values else 0.0
