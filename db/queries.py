import logging

from db import client

logger = logging.getLogger(__name__)


def upsert_kline(candle: dict):
    client.upsert("raw_klines", candle, on_conflict="open_time")


def upsert_depth_snapshot(snapshot: dict):
    client.insert("depth_snapshots", snapshot)


def insert_futures_snapshot(snapshot: dict):
    try:
        client.insert("futures_snapshots", snapshot)
    except Exception as e:
        logger.debug(f"futures_snapshots insert failed (table may not exist): {e}")


def insert_aggtrade_snapshot(snapshot: dict):
    try:
        client.insert("aggtrade_snapshots", snapshot)
    except Exception as e:
        logger.debug(f"aggtrade_snapshots insert failed (table may not exist): {e}")


def get_latest_klines(n: int) -> list[dict]:
    rows = client.select("raw_klines", params={
        "order": "open_time.desc",
        "limit": str(n),
    })
    return list(reversed(rows))  # oldest first


def get_depth_for_range(start_time: str, end_time: str) -> list[dict]:
    return client.select("depth_snapshots", params={
        "snapshot_time": f"gte.{start_time}",
        "order": "snapshot_time.asc",
    })


def insert_feature_row(row: dict):
    try:
        client.upsert("features", row, on_conflict="feature_time")
    except Exception as e:
        # If columns don't exist yet, strip unknown cols and retry with base columns only
        logger.warning(f"Feature upsert failed: {e}. Retrying with base columns...")
        base_cols = {
            "feature_time", "close_price", "open_price",
            "return_1m", "return_3m", "return_5m",
            "volume_ratio_20", "volume_spike",
            "atr_7", "bollinger_width",
            "ema_cross", "rsi_7",
            "order_flow_imbalance", "target",
            # Futures columns (may or may not exist)
            "funding_rate", "oi_change_pct", "long_short_ratio",
            "taker_buy_ratio", "whale_ratio", "whale_net_flow",
        }
        # Try progressively smaller sets — strip newest columns first
        new_cols = {"poly_up_price", "poly_spread", "price_acceleration", "atr_pct", "bollinger_pos"}
        futures_cols = {"funding_rate", "oi_change_pct", "long_short_ratio", "taker_buy_ratio", "whale_ratio", "whale_net_flow"}
        for attempt, skip_cols in enumerate([
            new_cols,
            new_cols | futures_cols,
            new_cols | futures_cols | {"open_price"},
        ]):
            try:
                slim_row = {k: v for k, v in row.items() if k not in skip_cols}
                client.upsert("features", slim_row, on_conflict="feature_time")
                logger.info(f"Feature upsert succeeded on retry (attempt {attempt + 1})")
                return
            except Exception:
                continue
        logger.error("Feature upsert failed on all retries")


def backfill_target(feature_time: str, target_value: int):
    client.update("features", {"target": target_value}, {"feature_time": feature_time})


def get_features_without_target(limit: int = 20) -> list[dict]:
    """Get recent features with NULL target (limited to avoid blocking)."""
    return client.select("features", params={
        "target": "is.null",
        "order": "feature_time.desc",
        "limit": str(limit),
    })


def get_training_data(start_time: str | None = None, end_time: str | None = None) -> list[dict]:
    """Fetch all training data with pagination (Supabase returns max 1000 per query)."""
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        params = {
            "target": "not.is.null",
            "order": "feature_time.asc",
            "limit": str(page_size),
            "offset": str(offset),
        }
        if start_time and end_time:
            params["and"] = f"(feature_time.gte.{start_time},feature_time.lte.{end_time})"
        elif start_time:
            params["feature_time"] = f"gte.{start_time}"
        elif end_time:
            params["feature_time"] = f"lte.{end_time}"

        rows = client.select("features", params=params)
        all_rows.extend(rows)

        if len(rows) < page_size:
            break
        offset += page_size

    return all_rows


def insert_prediction(pred: dict):
    client.insert("predictions", pred)


def get_latest_predictions(n: int) -> list[dict]:
    return client.select("predictions", params={
        "order": "predicted_at.desc",
        "limit": str(n),
    })


def get_predictions_without_outcome(limit: int = 20) -> list[dict]:
    """Get recent predictions with NULL actual_target."""
    return client.select("predictions", params={
        "actual_target": "is.null",
        "order": "predicted_at.desc",
        "limit": str(limit),
    })


def update_prediction_outcome(prediction_id: int, actual_target: int):
    client.update("predictions", {"actual_target": actual_target}, {"id": prediction_id})


def resolve_prediction_by_feature_time(feature_time: str, actual_target: int) -> bool:
    """Resolve a prediction by feature_time. Only updates if actual_target is still NULL."""
    try:
        rows = client.select("predictions", params={
            "feature_time": f"eq.{feature_time}",
            "actual_target": "is.null",
            "limit": "1",
        })
        if rows:
            client.update("predictions", {"actual_target": actual_target}, {"id": rows[0]["id"]})
            return True
    except Exception as e:
        logger.debug(f"resolve_prediction_by_feature_time failed: {e}")
    return False


def insert_model_metrics(metrics: dict):
    # Columns that may not exist in the DB schema yet
    _extra_cols = {
        "optimal_threshold", "optimal_threshold_up", "optimal_threshold_down",
        "model_type", "n_up_signals", "n_down_signals", "n_skip",
        "n_over_signals", "n_under_signals",
        "confusion_matrix", "true_positives", "false_positives",
        "true_negatives", "false_negatives",
    }
    try:
        client.insert("model_metrics", metrics)
    except Exception:
        safe = {k: v for k, v in metrics.items() if k not in _extra_cols}
        try:
            client.insert("model_metrics", safe)
        except Exception as e:
            logger.warning(f"model_metrics insert failed: {e}")


def get_model_metrics(model_version: str) -> list[dict]:
    return client.select("model_metrics", params={
        "model_version": f"eq.{model_version}",
        "order": "fold_index.asc",
    })


def get_latest_model_version() -> str | None:
    rows = client.select("model_metrics", columns="model_version", params={
        "order": "inserted_at.desc",
        "limit": "1",
    })
    if rows:
        return rows[0]["model_version"]
    return None


def insert_backtest_trades(trades: list[dict]):
    if trades:
        # Batch insert in chunks
        chunk_size = 100
        for i in range(0, len(trades), chunk_size):
            client.insert("backtest_trades", trades[i:i + chunk_size])


def get_backtest_trades(run_id: str) -> list[dict]:
    return client.select("backtest_trades", params={
        "backtest_run": f"eq.{run_id}",
        "order": "signal_time.asc",
    })


def get_latest_features() -> dict | None:
    rows = client.select("features", params={
        "order": "feature_time.desc",
        "limit": "1",
    })
    return rows[0] if rows else None


def get_latest_features_n(n: int) -> list[dict]:
    """Get the N most recent feature rows (for regime filter warmup)."""
    return client.select("features", params={
        "order": "feature_time.desc",
        "limit": str(n),
    })


def get_kline_at_time(target_time: str) -> dict | None:
    rows = client.select("raw_klines", params={
        "open_time": f"eq.{target_time}",
        "limit": "1",
    })
    return rows[0] if rows else None


def get_klines_at_times(times: list[str]) -> dict[str, dict]:
    """Batch fetch klines for multiple open_times. Returns {open_time: candle}."""
    if not times:
        return {}
    unique = list(set(times))
    csv = ",".join(f'"{t}"' for t in unique)
    rows = client.select("raw_klines", columns="open_time,open,close", params={
        "open_time": f"in.({csv})",
    })
    return {r["open_time"]: r for r in rows}


def get_all_klines_range(start_time: str, end_time: str) -> list[dict]:
    return client.select("raw_klines", params={
        "open_time": f"gte.{start_time}",
        "order": "open_time.asc",
    })
