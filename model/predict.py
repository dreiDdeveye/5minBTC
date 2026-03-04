import json
import logging
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

import config
from db import queries

logger = logging.getLogger(__name__)

_model: xgb.XGBClassifier | None = None
_model_version: str | None = None
_threshold: float = config.SIGNAL_THRESHOLD
_active_cols: list[str] | None = None


def load_latest_model():
    global _model, _model_version, _threshold, _active_cols
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = sorted(
        [f for f in config.MODEL_DIR.glob("xgb_*.json") if "_meta" not in f.name],
        reverse=True,
    )
    if not artifacts:
        logger.warning("No trained model found")
        return False
    _model = xgb.XGBClassifier()
    _model.load_model(str(artifacts[0]))
    _model_version = artifacts[0].stem
    logger.info(f"Loaded model: {_model_version}")

    # Load metadata (threshold + active columns)
    meta_path = artifacts[0].with_name(f"{_model_version}_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            _threshold = meta.get("threshold", config.SIGNAL_THRESHOLD)
            _active_cols = meta.get("active_cols")
            logger.info(
                f"Threshold: {_threshold:.3f}, "
                f"features: {len(_active_cols) if _active_cols else '?'}"
            )
        except Exception as e:
            logger.warning(f"Failed to load model metadata: {e}")
            _threshold = config.SIGNAL_THRESHOLD
            _active_cols = None
    else:
        _threshold = config.SIGNAL_THRESHOLD
        _active_cols = None

    return True


async def predict_latest(feature_row: dict) -> dict | None:
    global _model, _model_version
    if _model is None:
        if not load_latest_model():
            return None

    # Determine which columns to use
    if _active_cols:
        cols = _active_cols
    else:
        model_n = _model.n_features_in_
        cols = config.FEATURE_COLS[:model_n] if model_n < len(config.FEATURE_COLS) else config.FEATURE_COLS

    feature_vec = np.array([[
        int(feature_row[col]) if isinstance(feature_row.get(col), bool)
        else float(feature_row.get(col, 0) or 0)
        for col in cols
    ]])

    probability = float(_model.predict_proba(feature_vec)[0, 1])

    # Binary signal: OVER (1) or UNDER (-1)
    signal = 1 if probability > _threshold else -1

    signal_name = "OVER" if signal == 1 else "UNDER"
    logger.info(f"Prediction: P={probability:.3f} Signal={signal_name}")

    db_pred = {
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "feature_time": feature_row["feature_time"],
        "model_version": _model_version,
        "probability": probability,
        "signal": signal,
    }
    queries.insert_prediction(db_pred)
    db_pred["close_price"] = feature_row["close_price"]
    return db_pred
