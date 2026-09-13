import numpy as np
import pandas as pd
import pytest

from common.classifier_utils import (
    align_train_test,
    binary_classification_metrics,
    calibration_table,
    clean_and_encode,
    feature_columns,
    multiclass_top3_metrics,
    top10_cut_precision_recall,
    top3_accuracy_by_race,
)


def test_clean_and_encode_drops_leakage_and_imputes_and_encodes():
    df = pd.DataFrame(
        {
            "season": [2023, 2023],
            "leak_col": [1, 2],
            "numeric_feat": [10.0, None],
            "team": ["Red Bull", "Mercedes"],
        }
    )
    result = clean_and_encode(df, drop_cols=["leak_col"])
    assert "leak_col" not in result.columns
    assert result["numeric_feat"].isna().sum() == 0
    assert result.loc[0, "numeric_feat"] == 10.0
    assert (
        result.loc[1, "numeric_feat"] == 10.0
    )  # median-imputed from the one known value
    assert "team_Red Bull" in result.columns
    assert "team_Mercedes" in result.columns


def test_align_train_test_fills_missing_dummy_columns_with_zero():
    train = pd.DataFrame({"a": [1, 2], "team_Ferrari": [1, 0]})
    test = pd.DataFrame({"a": [3]})
    aligned_train, aligned_test = align_train_test(train, test)
    assert "team_Ferrari" in aligned_test.columns
    assert aligned_test.loc[0, "team_Ferrari"] == 0


def test_feature_columns_excludes_identifiers():
    df = pd.DataFrame({"season": [1], "round": [1], "feat_a": [1], "label": [0]})
    cols = feature_columns(df, non_feature_cols=["season", "round", "label"])
    assert cols == ["feat_a"]


def test_binary_classification_metrics_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.95, 0.05, 0.9, 0.1])
    metrics = binary_classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["log_loss"] < 0.2
    assert metrics["brier_score"] < 0.02


def test_calibration_table_has_expected_columns_and_row_count():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45])
    table = calibration_table(y_true, y_prob, n_buckets=2)
    assert {"bucket", "predicted_mean", "actual_rate", "n"}.issubset(table.columns)
    assert table["n"].sum() == 10


def test_multiclass_top3_metrics_returns_expected_keys():
    y_true_class = np.array([1, 2, 3, 0])
    classes = [0, 1, 2, 3]
    proba = np.array(
        [
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
            [0.7, 0.1, 0.1, 0.1],
        ]
    )
    metrics = multiclass_top3_metrics(y_true_class, proba, classes)
    assert set(metrics) == {
        "win_log_loss",
        "win_brier_score",
        "top3_log_loss",
        "top3_brier_score",
    }
    assert metrics["win_brier_score"] < 0.1


def test_top3_accuracy_by_race_perfect_pick():
    race_keys = pd.DataFrame({"season": [2025] * 4, "round": [1] * 4})
    y_true_class = np.array([1, 2, 3, 0])
    p_win = np.array(
        [0.9, 0.6, 0.5, 0.1]
    )  # top-3 by p_win = indices 0,1,2 = classes 1,2,3
    accuracy = top3_accuracy_by_race(race_keys, y_true_class, p_win)
    assert accuracy == pytest.approx(1.0)


def test_top10_cut_precision_recall_matches_hand_computed_value():
    # Two races of 4 drivers each; k = number of true positives in that race.
    # Race 1: k=2 (rows 0,1 true). Top-2 by prob = rows 0,2 -> 1/2 correct.
    # Race 2: k=2 (rows 4,5 true). Top-2 by prob = rows 4,5 -> 2/2 correct.
    # Mean across races = (0.5 + 1.0) / 2 = 0.75.
    race_keys = pd.DataFrame({"season": [2025] * 8, "round": [1, 1, 1, 1, 2, 2, 2, 2]})
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1, 0.9, 0.8, 0.2, 0.1])

    result = top10_cut_precision_recall(race_keys, y_true, y_prob)
    assert result["top10_cut_precision"] == pytest.approx(0.75)
    # Same selection rule as Phase 5: picking exactly k makes precision == recall.
    assert result["top10_cut_recall"] == pytest.approx(0.75)


def test_top10_cut_precision_recall_skips_races_with_no_positives():
    race_keys = pd.DataFrame({"season": [2025] * 4, "round": [1, 1, 2, 2]})
    y_true = np.array([1, 0, 0, 0])  # round 2 has no true top-10 finishers
    y_prob = np.array([0.9, 0.1, 0.7, 0.3])

    result = top10_cut_precision_recall(race_keys, y_true, y_prob)
    assert result["top10_cut_precision"] == pytest.approx(1.0)


def test_top10_cut_precision_recall_all_races_empty_returns_nan():
    race_keys = pd.DataFrame({"season": [2025] * 2, "round": [1, 1]})
    result = top10_cut_precision_recall(
        race_keys, np.array([0, 0]), np.array([0.4, 0.6])
    )
    assert np.isnan(result["top10_cut_precision"])
