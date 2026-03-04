import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

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


def _find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find single threshold that maximizes precision for OVER signals."""
    best_t = config.SIGNAL_THRESHOLD
    best_score = 0.0
    for t in np.arange(0.45, 0.65, 0.01):
        y_pred = (y_prob > t).astype(int)
        if y_pred.sum() < 5:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec > best_score:
            best_score = prec
            best_t = float(t)
    return best_t


def _drop_zero_variance(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Drop feature columns with zero variance (constant values). Returns active cols."""
    active = []
    dropped = []
    for col in feature_cols:
        if col in df.columns and df[col].std() > 1e-10:
            active.append(col)
        else:
            dropped.append(col)
    if dropped:
        logger.info(f"Dropped zero-variance features: {dropped}")
    return active


def _build_model():
    """Create an XGBClassifier with config hyperparameters."""
    return xgb.XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE,
        min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
        eval_metric="logloss",
        random_state=42,
    )


def _walk_forward_model(
    df: pd.DataFrame, active_cols: list[str], model_type: str = "xgb"
) -> tuple[list[dict], list[float]]:
    """Run walk-forward validation for a specific model type.
    Returns (fold_metrics, fold_thresholds).
    """
    n = len(df)
    fold_metrics = []
    fold_thresholds = []
    train_end = config.WALK_FORWARD_TRAIN_SIZE
    fold_idx = 0

    while train_end + config.WALK_FORWARD_TEST_SIZE <= n:
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + config.WALK_FORWARD_TEST_SIZE]

        X_train = train_df[active_cols].values
        y_train = train_df["target"].values
        X_test = test_df[active_cols].values
        y_test = test_df["target"].values

        if model_type == "xgb":
            model = _build_model()
            model.set_params(scale_pos_weight=_scale_pos_weight(y_train))
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
            model.fit(X_train_s, y_train)
            y_prob = model.predict_proba(X_test_s)[:, 1]

        # Find optimal threshold for this fold
        opt_t = _find_optimal_threshold(y_test, y_prob)
        fold_thresholds.append(opt_t)

        # Evaluate
        y_pred = (y_prob > opt_t).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics.update({
            "fold_index": fold_idx,
            "model_type": model_type,
            "train_start": str(train_df.iloc[0]["feature_time"]),
            "train_end": str(train_df.iloc[-1]["feature_time"]),
            "test_start": str(test_df.iloc[0]["feature_time"]),
            "test_end": str(test_df.iloc[-1]["feature_time"]),
            "n_samples_train": len(train_df),
            "n_samples_test": len(test_df),
            "optimal_threshold": opt_t,
        })

        # Signal counts
        n_over = int(np.sum(y_prob > opt_t))
        n_under = len(y_test) - n_over
        metrics["n_over_signals"] = n_over
        metrics["n_under_signals"] = n_under

        fold_metrics.append(metrics)
        logger.info(
            f"  [{model_type}] Fold {fold_idx}: acc={metrics['accuracy']:.3f} "
            f"F1={metrics['f1']:.3f} threshold={opt_t:.2f} "
            f"signals={n_over}OVER/{n_under}UNDER"
        )

        train_end += config.WALK_FORWARD_TEST_SIZE
        fold_idx += 1

    return fold_metrics, fold_thresholds


def _log_model_comparison(xgb_folds: list[dict], logreg_folds: list[dict]):
    """Log side-by-side comparison of XGBoost vs LogReg."""
    logger.info("=== MODEL COMPARISON ===")
    for metric in ["accuracy", "precision_score", "recall", "f1", "roc_auc"]:
        xgb_vals = [f.get(metric, 0) or 0 for f in xgb_folds]
        lr_vals = [f.get(metric, 0) or 0 for f in logreg_folds]
        xgb_avg = np.mean(xgb_vals) if xgb_vals else 0
        lr_avg = np.mean(lr_vals) if lr_vals else 0
        winner = "XGB" if xgb_avg >= lr_avg else "LogReg"
        logger.info(f"  {metric}: XGB={xgb_avg:.3f} LogReg={lr_avg:.3f} [{winner}]")


def walk_forward_train(features_df: pd.DataFrame | None = None) -> tuple:
    """
    Walk-forward validation with expanding window.
    Trains both XGBoost and Logistic Regression, compares them.
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
    if "volume_spike" in features_df.columns:
        features_df["volume_spike"] = features_df["volume_spike"].astype(int)

    n = len(features_df)
    logger.info(f"Training data: {n} rows")

    # Drop zero-variance features
    active_cols = _drop_zero_variance(features_df, config.FEATURE_COLS)
    active_cols = [c for c in active_cols if c in features_df.columns]
    logger.info(f"Active features ({len(active_cols)}): {active_cols}")

    if n < config.WALK_FORWARD_TRAIN_SIZE + config.WALK_FORWARD_TEST_SIZE:
        logger.warning(f"Not enough data for walk-forward. Training on all {n} rows.")
        X = features_df[active_cols].values
        y = features_df["target"].values

        model = _build_model()
        model.set_params(scale_pos_weight=_scale_pos_weight(y))
        model.fit(X, y, verbose=False)

        importance = dict(zip(active_cols, [float(v) for v in model.feature_importances_]))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Feature importance: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted_imp[:10]))

        version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(config.MODEL_DIR / f"{version}.json"))

        meta = {
            "active_cols": active_cols,
            "threshold": config.SIGNAL_THRESHOLD,
            "importance": importance,
        }
        (config.MODEL_DIR / f"{version}_meta.json").write_text(json.dumps(meta))

        logger.info(f"Model saved: {version}")
        return model, [], version

    # --- Walk-forward for XGBoost ---
    logger.info("--- XGBoost Walk-Forward ---")
    xgb_folds, xgb_thresholds = _walk_forward_model(features_df, active_cols, "xgb")

    # --- Walk-forward for Logistic Regression ---
    logger.info("--- Logistic Regression Walk-Forward ---")
    logreg_folds, logreg_thresholds = _walk_forward_model(features_df, active_cols, "logreg")

    # --- Compare models ---
    _log_model_comparison(xgb_folds, logreg_folds)

    # --- Train final XGBoost on all data (production model) ---
    X_all = features_df[active_cols].values
    y_all = features_df["target"].values

    final_model = _build_model()
    final_model.set_params(scale_pos_weight=_scale_pos_weight(y_all))
    final_model.fit(X_all, y_all, verbose=False)

    # Calibrated threshold = average of fold-optimal thresholds
    cal_t = float(np.mean(xgb_thresholds)) if xgb_thresholds else config.SIGNAL_THRESHOLD
    logger.info(f"Calibrated threshold: {cal_t:.3f}")

    # Feature importance from final model
    importance = dict(zip(active_cols, [float(v) for v in final_model.feature_importances_]))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Feature importance: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted_imp))

    version = f"xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(config.MODEL_DIR / f"{version}.json"))

    # Save metadata with threshold and LogReg comparison
    logreg_summary = {
        "avg_accuracy": float(np.mean([f["accuracy"] for f in logreg_folds])),
        "avg_f1": float(np.mean([f["f1"] for f in logreg_folds])),
        "avg_precision": float(np.mean([f["precision_score"] for f in logreg_folds])),
        "avg_roc_auc": float(np.mean([f.get("roc_auc", 0) or 0 for f in logreg_folds])),
    }

    meta = {
        "active_cols": active_cols,
        "threshold": cal_t,
        "importance": importance,
        "n_folds": len(xgb_folds),
        "fold_thresholds": xgb_thresholds,
        "logreg_comparison": logreg_summary,
    }
    (config.MODEL_DIR / f"{version}_meta.json").write_text(json.dumps(meta))

    # Store XGB fold metrics in DB
    fold_metrics = xgb_folds
    for m in fold_metrics:
        m["model_version"] = version
        queries.insert_model_metrics(m)

    logger.info(f"Model saved: {version} ({len(xgb_folds)} folds)")
    return final_model, fold_metrics, version
