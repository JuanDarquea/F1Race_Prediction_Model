"""Shared feature prep and metrics helpers for Phase 10 classifiers."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clean_and_encode(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """Drop leakage columns, median-impute numeric NaNs, one-hot encode categoricals."""
    feature_df = df.copy()
    for col in drop_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        median_value = feature_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        feature_df[col] = feature_df[col].fillna(median_value)

    categorical_cols = [
        col
        for col in ["driver_name", "team", "track_type", "track"]
        if col in feature_df.columns
    ]
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)

    object_cols = feature_df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        feature_df = feature_df.drop(columns=list(object_cols))

    return feature_df


def align_train_test(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align one-hot columns between a train and test split (fills missing with 0)."""
    return train.align(test, join="left", axis=1, fill_value=0)


def feature_columns(df: pd.DataFrame, non_feature_cols: List[str]) -> List[str]:
    return [col for col in df.columns if col not in non_feature_cols]


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Log-loss, Brier score, ROC-AUC, precision/recall at a 0.5 threshold."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_buckets: int = 5
) -> pd.DataFrame:
    """Bucket predictions by predicted probability and compare to the actual rate per bucket."""
    frame = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    frame["bucket"] = pd.qcut(frame["y_prob"], q=n_buckets, duplicates="drop")
    table = frame.groupby("bucket", observed=True).agg(
        predicted_mean=("y_prob", "mean"),
        actual_rate=("y_true", "mean"),
        n=("y_true", "size"),
    )
    return table.reset_index()


def multiclass_top3_metrics(
    y_true_class: np.ndarray, proba: np.ndarray, classes: List[int]
) -> dict:
    """Log-loss/Brier for the P(win) and P(top3) views of a 4-class (1st/2nd/3rd/other) model."""
    y_true_win = (y_true_class == 1).astype(int)
    win_idx = classes.index(1)
    p_win = proba[:, win_idx]

    y_true_top3 = (y_true_class != 0).astype(int)
    top3_idx = [classes.index(c) for c in (1, 2, 3) if c in classes]
    p_top3 = proba[:, top3_idx].sum(axis=1)

    return {
        "win_log_loss": float(log_loss(y_true_win, p_win, labels=[0, 1])),
        "win_brier_score": float(brier_score_loss(y_true_win, p_win)),
        "top3_log_loss": float(log_loss(y_true_top3, p_top3, labels=[0, 1])),
        "top3_brier_score": float(brier_score_loss(y_true_top3, p_top3)),
    }


def top3_accuracy_by_race(
    race_keys: pd.DataFrame, y_true_class: np.ndarray, p_win: np.ndarray
) -> float:
    """Fraction of the actual podium (classes 1/2/3) that appear in the model's
    top-3 picks by predicted P(win), averaged across races."""
    frame = race_keys.reset_index(drop=True).copy()
    frame["y_true_class"] = y_true_class
    frame["p_win"] = p_win
    hits = []
    for _, group in frame.groupby(["season", "round"]):
        actual_podium = set(group.index[group["y_true_class"].isin([1, 2, 3])])
        predicted_top3 = set(group.nlargest(3, "p_win").index)
        if not actual_podium:
            continue
        hits.append(len(actual_podium & predicted_top3) / len(actual_podium))
    return float(np.mean(hits)) if hits else float("nan")
