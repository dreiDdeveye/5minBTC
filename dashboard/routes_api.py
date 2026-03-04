import asyncio
import json

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from db import queries
import config
import live_state

router = APIRouter()


@router.get("/candles")
def get_candles(limit: int = 200):
    return queries.get_latest_klines(limit)


@router.get("/predictions")
def get_predictions(limit: int = 50):
    preds = queries.get_latest_predictions(limit)
    # Batch fetch all needed prices in one query instead of N queries
    feature_times = [p["feature_time"] for p in preds if p.get("feature_time")]
    kline_map = queries.get_klines_at_times(feature_times) if feature_times else {}
    for p in preds:
        if p.get("feature_time"):
            candle = kline_map.get(p["feature_time"])
            if candle:
                p.setdefault("close_price", candle["close"])
                p["open_price"] = candle["open"]  # PTB baseline
    return preds


@router.get("/predictions/live")
def get_live_prediction():
    preds = queries.get_latest_predictions(1)
    if preds:
        pred = preds[0]
        if pred.get("feature_time"):
            candle = queries.get_kline_at_time(pred["feature_time"])
            if candle:
                pred["close_price"] = candle["close"]
                pred["open_price"] = candle["open"]  # PTB baseline
        return pred
    return {"probability": None, "signal": None, "model_version": None, "close_price": None, "open_price": None}


@router.get("/metrics/latest")
def get_latest_metrics():
    version = queries.get_latest_model_version()
    if not version:
        return {"error": "No model metrics found"}
    return {
        "model_version": version,
        "folds": queries.get_model_metrics(version),
    }


@router.get("/metrics/{model_version}")
def get_metrics(model_version: str):
    return queries.get_model_metrics(model_version)


@router.get("/backtest/{run_id}")
def get_backtest(run_id: str):
    return queries.get_backtest_trades(run_id)


@router.get("/features/latest")
def get_latest_features():
    """Get the most recent feature row."""
    rows = queries.get_latest_features()
    return rows if rows else {}


@router.get("/status")
def get_status():
    candles = queries.get_latest_klines(1)
    preds = queries.get_latest_predictions(1)
    return {
        "last_candle": candles[0]["close_time"] if candles else None,
        "last_prediction": preds[0]["predicted_at"] if preds else None,
        "current_price": live_state.price or (candles[0]["close"] if candles else None),
    }


@router.get("/price")
def get_price():
    return live_state.to_dict()


@router.get("/polymarket")
def get_polymarket():
    """Live Polymarket odds for current BTC 5-min window."""
    return live_state.polymarket


@router.post("/predictions/resolve")
async def resolve_prediction(body: dict):
    """Resolve a prediction outcome immediately when the 5-min window expires.
    Called by the frontend so the W/L is stored permanently in the DB."""
    feature_time = body.get("feature_time")
    actual_target = body.get("actual_target")
    if feature_time is None or actual_target is None:
        return {"ok": False, "error": "missing fields"}
    ok = queries.resolve_prediction_by_feature_time(feature_time, int(actual_target))
    return {"ok": ok}


@router.get("/ptb")
def get_ptb(start_time: int):
    """Get the PTB (Price To Beat) by fetching the 1m candle open from Binance REST API.
    start_time is the 5-min window start in milliseconds."""
    try:
        resp = httpx.get(
            f"{config.BINANCE_REST_BASE}/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "startTime": start_time, "limit": 1},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return {"open": float(data[0][1]), "time": data[0][0]}
    except Exception:
        pass
    return {"open": None, "time": None}


@router.websocket("/ws/price")
async def ws_price(websocket: WebSocket):
    """Real-time price WebSocket. Pushes every update from Binance."""
    await websocket.accept()
    queue = live_state.subscribe()
    try:
        while True:
            data = await queue.get()
            await websocket.send_json(data)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        live_state.unsubscribe(queue)
