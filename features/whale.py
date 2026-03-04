"""Features derived from aggregated trade whale detection."""


def whale_ratio_feature(aggtrade_snapshots: list[dict]) -> float:
    """Average whale volume ratio across snapshots."""
    if not aggtrade_snapshots:
        return 0.0
    values = [s.get("whale_ratio", 0) for s in aggtrade_snapshots]
    return sum(values) / len(values)


def whale_net_flow_feature(aggtrade_snapshots: list[dict]) -> float:
    """Average whale net flow (buy-sell imbalance) across snapshots."""
    if not aggtrade_snapshots:
        return 0.0
    values = [s.get("whale_net_flow", 0) for s in aggtrade_snapshots]
    return sum(values) / len(values)
