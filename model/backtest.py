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


def run_backtest() -> dict:
    """Run full walk-forward backtest and store results."""
    data = queries.get_training_data()
    if not data:
        logger.error("No training data for backtest")
        return {"error": "No data"}

    df = pd.DataFrame(data).sort_values("feature_time").reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    df["volume_spike"] = df["volume_spike"].astype(int)

    n = len(df)
    train_size = config.WALK_FORWARD_TRAIN_SIZE
    test_size = config.WALK_FORWARD_TEST_SIZE

    if n < train_size + test_size:
        logger.error(f"Not enough data: {n} rows (need {train_size + test_size})")
        return {"error": "Not enough data"}

    run_id = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    all_trades = []
    all_metrics = []
    train_end = train_size
    fold_idx = 0

    while train_end + test_size <= n:
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + test_size]

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
        model.fit(X_train, y_train, verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > config.SIGNAL_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics.update({
            "model_version": run_id,
            "fold_index": fold_idx,
            "train_start": str(train_df.iloc[0]["feature_time"]),
            "train_end": str(train_df.iloc[-1]["feature_time"]),
            "test_start": str(test_df.iloc[0]["feature_time"]),
            "test_end": str(test_df.iloc[-1]["feature_time"]),
            "n_samples_train": len(train_df),
            "n_samples_test": len(test_df),
        })
        all_metrics.append(metrics)

        # Record individual trades
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            all_trades.append({
                "backtest_run": run_id,
                "signal_time": str(row["feature_time"]),
                "entry_price": float(row["close_price"]),
                "exit_price": float(row["close_price"]),  # simplified
                "probability": float(y_prob[i]),
                "signal": int(y_pred[i]),
                "actual_target": int(y_test[i]),
                "correct": bool(y_pred[i] == y_test[i]),
            })

        train_end += test_size
        fold_idx += 1

    # Store results
    for m in all_metrics:
        queries.insert_model_metrics(m)
    queries.insert_backtest_trades(all_trades)

    # Summary
    signaled = [t for t in all_trades if t["signal"] == 1]
    correct_signals = [t for t in signaled if t["correct"]]
    hit_rate = len(correct_signals) / len(signaled) if signaled else 0

    summary = {
        "run_id": run_id,
        "total_samples": len(all_trades),
        "total_signals": len(signaled),
        "correct_signals": len(correct_signals),
        "hit_rate": hit_rate,
        "folds": fold_idx,
        "avg_accuracy": np.mean([m["accuracy"] for m in all_metrics]),
    }
    logger.info(f"Backtest {run_id}: {fold_idx} folds, hit_rate={hit_rate:.3f}")
    return summary
