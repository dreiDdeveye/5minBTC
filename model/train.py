import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from db import queries
from model.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def walk_forward_train(features_df: pd.DataFrame | None = None) -> tuple:
    """
    Walk-forward validation with expanding window.
    Returns (final_model, fold_metrics, model_version).
    """
    if features_df is None:
        data = queries.get_training_data()
        if not data:
            logger.error("No training data available")
            return None, [], None
        features_df = pd.DataFrame(data)

    features_df = features_df.sort_values("feature_time").reset_index(drop=True)

    # Filter rows with target
    features_df = features_df.dropna(subset=["target"])
    features_df["target"] = features_df["target"].astype(int)
    features_df["volume_spike"] = features_df["volume_spike"].astype(int)

    n = len(features_df)
    logger.info(f"Training data: {n} rows")

    if n < config.WALK_FORWARD_TRAIN_SIZE + config.WALK_FORWARD_TEST_SIZE:
        logger.warning(f"Not enough data for walk-forward. Training on all {n} rows.")
        # Just train on everything
        X = features_df[config.FEATURE_COLS].values
        y = features_df["target"].values

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=_scale_pos_weight(y),
            eval_metric="logloss", random_state=42,
        )
        model.fit(X, y, verbose=False)

        version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(config.MODEL_DIR / f"{version}.json"))
        logger.info(f"Model saved: {version}")
        return model, [], version

    # Walk-forward
    fold_metrics = []
    train_end = config.WALK_FORWARD_TRAIN_SIZE
    fold_idx = 0

    while train_end + config.WALK_FORWARD_TEST_SIZE <= n:
        train_df = features_df.iloc[:train_end]
        test_df = features_df.iloc[train_end:train_end + config.WALK_FORWARD_TEST_SIZE]

        X_train = train_df[config.FEATURE_COLS].values
        y_train = train_df["target"].values
        X_test = test_df[config.FEATURE_COLS].values
        y_test = test_df["target"].values

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=_scale_pos_weight(y_train),
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > config.SIGNAL_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics.update({
            "fold_index": fold_idx,
            "train_start": str(train_df.iloc[0]["feature_time"]),
            "train_end": str(train_df.iloc[-1]["feature_time"]),
            "test_start": str(test_df.iloc[0]["feature_time"]),
            "test_end": str(test_df.iloc[-1]["feature_time"]),
            "n_samples_train": len(train_df),
            "n_samples_test": len(test_df),
        })
        fold_metrics.append(metrics)
        logger.info(f"Fold {fold_idx}: accuracy={metrics['accuracy']:.3f} F1={metrics['f1']:.3f}")

        train_end += config.WALK_FORWARD_TEST_SIZE
        fold_idx += 1

    # Final model on all data
    X_all = features_df[config.FEATURE_COLS].values
    y_all = features_df["target"].values

    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=_scale_pos_weight(y_all),
        eval_metric="logloss", random_state=42,
    )
    final_model.fit(X_all, y_all, verbose=False)

    version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(config.MODEL_DIR / f"{version}.json"))

    # Store metrics in DB
    for m in fold_metrics:
        m["model_version"] = version
        queries.insert_model_metrics(m)

    logger.info(f"Model saved: {version} ({fold_idx} folds)")
    return final_model, fold_metrics, version
