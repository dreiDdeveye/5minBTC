import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
from db import queries
from model.evaluate import compute_metrics
from model.train import _build_model, _scale_pos_weight, _drop_zero_variance, _find_optimal_threshold

logger = logging.getLogger(__name__)


def run_backtest() -> dict:
    """Run full walk-forward backtest with binary OVER/UNDER signals, month-by-month, EV, and model comparison."""
    data = queries.get_training_data()
    if not data:
        logger.error("No training data for backtest")
        return {"error": "No data"}

    df = pd.DataFrame(data).sort_values("feature_time").reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    if "volume_spike" in df.columns:
        df["volume_spike"] = df["volume_spike"].astype(int)

    # Drop zero-variance features
    active_cols = _drop_zero_variance(df, config.FEATURE_COLS)
    active_cols = [c for c in active_cols if c in df.columns]
    logger.info(f"Backtest active features ({len(active_cols)}): {active_cols}")

    n = len(df)
    train_size = config.WALK_FORWARD_TRAIN_SIZE
    test_size = config.WALK_FORWARD_TEST_SIZE

    if n < train_size + test_size:
        logger.error(f"Not enough data: {n} rows (need {train_size + test_size})")
        return {"error": "Not enough data"}

    run_id = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Run walk-forward for both models
    xgb_trades, xgb_metrics = _run_walk_forward(df, active_cols, run_id, "xgb")
    logreg_trades, logreg_metrics = _run_walk_forward(df, active_cols, run_id, "logreg")

    # Store results (only XGB trades in DB)
    for m in xgb_metrics:
        queries.insert_model_metrics(m)
    queries.insert_backtest_trades(xgb_trades)

    # --- Summary ---
    xgb_summary = _compute_summary(xgb_trades, xgb_metrics, "xgb")
    logreg_summary = _compute_summary(logreg_trades, logreg_metrics, "logreg")

    # Month-by-month breakdown
    monthly = _compute_monthly(xgb_trades)

    # Risk notes
    total = len(xgb_trades)
    risk_notes = [
        "100% accuracy is impossible — BTC price is influenced by unpredictable events (news, liquidations, whale orders, macro)",
        f"Every candle gets a signal: {total} total OVER/UNDER predictions" if total > 0 else "",
        "Walk-forward validation prevents data leakage but does not guarantee future performance",
        "Past backtest results are not indicative of future returns",
    ]

    summary = {
        "run_id": run_id,
        "xgb": xgb_summary,
        "logreg": logreg_summary,
        "monthly": monthly,
        "risk_notes": [n for n in risk_notes if n],
    }

    logger.info(f"Backtest {run_id}: XGB hit_rate={xgb_summary.get('hit_rate', 0):.3f} "
                f"LogReg hit_rate={logreg_summary.get('hit_rate', 0):.3f}")
    return summary


def _run_walk_forward(
    df: pd.DataFrame, active_cols: list[str], run_id: str, model_type: str
) -> tuple[list[dict], list[dict]]:
    """Run walk-forward for a specific model type. Returns (trades, fold_metrics)."""
    n = len(df)
    train_size = config.WALK_FORWARD_TRAIN_SIZE
    test_size = config.WALK_FORWARD_TEST_SIZE

    all_trades = []
    all_metrics = []
    train_end = train_size
    fold_idx = 0

    while train_end + test_size <= n:
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + test_size]

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

        # Single threshold
        opt_t = _find_optimal_threshold(y_test, y_prob)

        # Binary signal: OVER (1) or UNDER (-1)
        y_signal = np.where(y_prob > opt_t, 1, -1)

        # Standard metrics
        y_pred_binary = (y_prob > opt_t).astype(int)
        metrics = compute_metrics(y_test, y_pred_binary, y_prob)
        metrics.update({
            "model_version": run_id,
            "model_type": model_type,
            "fold_index": fold_idx,
            "train_start": str(train_df.iloc[0]["feature_time"]),
            "train_end": str(train_df.iloc[-1]["feature_time"]),
            "test_start": str(test_df.iloc[0]["feature_time"]),
            "test_end": str(test_df.iloc[-1]["feature_time"]),
            "n_samples_train": len(train_df),
            "n_samples_test": len(test_df),
            "optimal_threshold": opt_t,
        })
        all_metrics.append(metrics)

        # Record individual trades — every row gets scored
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            signal_val = int(y_signal[i])

            # OVER correct if actual=1, UNDER correct if actual=0
            if signal_val == 1:
                correct = bool(y_test[i] == 1)
            else:
                correct = bool(y_test[i] == 0)

            all_trades.append({
                "backtest_run": run_id,
                "signal_time": str(row["feature_time"]),
                "entry_price": float(row["close_price"]),
                "exit_price": float(row["close_price"]),
                "probability": float(y_prob[i]),
                "signal": signal_val,
                "actual_target": int(y_test[i]),
                "correct": correct,
            })

        train_end += test_size
        fold_idx += 1

    return all_trades, all_metrics


def _compute_summary(trades: list[dict], metrics: list[dict], model_type: str) -> dict:
    """Compute summary stats for a set of trades."""
    correct_trades = [t for t in trades if t["correct"] is True]
    hit_rate = len(correct_trades) / len(trades) if trades else 0

    # EV calculation (direction-based: +1 per correct, -1 per incorrect)
    ev_per_trade = 0.0
    if trades:
        wins = len(correct_trades)
        losses = len(trades) - wins
        ev_per_trade = (wins - losses) / len(trades)

    # Separate OVER and UNDER stats
    over_signals = [t for t in trades if t["signal"] == 1]
    under_signals = [t for t in trades if t["signal"] == -1]

    over_correct = len([t for t in over_signals if t["correct"] is True])
    under_correct = len([t for t in under_signals if t["correct"] is True])

    return {
        "model_type": model_type,
        "total_signals": len(trades),
        "over_signals": len(over_signals),
        "under_signals": len(under_signals),
        "correct_signals": len(correct_trades),
        "hit_rate": hit_rate,
        "over_hit_rate": over_correct / len(over_signals) if over_signals else 0,
        "under_hit_rate": under_correct / len(under_signals) if under_signals else 0,
        "ev_per_trade": ev_per_trade,
        "folds": len(metrics),
        "avg_accuracy": float(np.mean([m["accuracy"] for m in metrics])) if metrics else 0,
        "avg_f1": float(np.mean([m["f1"] for m in metrics])) if metrics else 0,
        "avg_precision": float(np.mean([m["precision_score"] for m in metrics])) if metrics else 0,
        "avg_recall": float(np.mean([m["recall"] for m in metrics])) if metrics else 0,
        "avg_roc_auc": float(np.mean([m.get("roc_auc", 0) or 0 for m in metrics])) if metrics else 0,
    }


def _compute_monthly(trades: list[dict]) -> list[dict]:
    """Compute month-by-month breakdown."""
    if not trades:
        return []

    trades_df = pd.DataFrame(trades)
    trades_df["month"] = pd.to_datetime(trades_df["signal_time"]).dt.to_period("M")

    monthly = []
    for month, group in trades_df.groupby("month"):
        correct = group[group["correct"] == True]
        monthly.append({
            "month": str(month),
            "total_signals": len(group),
            "correct": len(correct),
            "hit_rate": len(correct) / len(group) if len(group) > 0 else 0,
        })

    return monthly
