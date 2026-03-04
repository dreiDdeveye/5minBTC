import logging
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

import config
from db import queries

logger = logging.getLogger(__name__)

_model: xgb.XGBClassifier | None = None
_model_version: str | None = None


def load_latest_model():
    global _model, _model_version
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = sorted(config.MODEL_DIR.glob("xgb_*.json"), reverse=True)
    if not artifacts:
        logger.warning("No trained model found")
        return False
    _model = xgb.XGBClassifier()
    _model.load_model(str(artifacts[0]))
    _model_version = artifacts[0].stem
    logger.info(f"Loaded model: {_model_version}")
    return True


async def predict_latest(feature_row: dict) -> dict | None:
    global _model, _model_version
    if _model is None:
        if not load_latest_model():
            return None

    feature_vec = np.array([[
        int(feature_row[col]) if isinstance(feature_row.get(col), bool)
        else feature_row[col]
        for col in config.FEATURE_COLS
    ]])

    probability = float(_model.predict_proba(feature_vec)[0, 1])
    signal = 1 if probability > config.SIGNAL_THRESHOLD else 0

    db_pred = {
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "feature_time": feature_row["feature_time"],
        "model_version": _model_version,
        "probability": probability,
        "signal": signal,
    }
    queries.insert_prediction(db_pred)
    # Return with close_price for API use (not stored in DB)
    db_pred["close_price"] = feature_row["close_price"]
    return db_pred
