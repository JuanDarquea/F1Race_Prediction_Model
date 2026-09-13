import pandas as pd

from phase11_classifier_evaluation import (
    build_comparison_table,
    load_regression_top10_baseline,
)


def test_load_regression_top10_baseline_picks_lowest_mae(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["xgboost", "random_forest"],
            "target": ["race", "race"],
            "mae": [2.4, 2.8],
            "top10_precision": [0.825, 0.817],
            "top10_recall": [0.825, 0.817],
        }
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    baseline = load_regression_top10_baseline(summary_path, "race")
    assert baseline["baseline_model"] == "xgboost"
    assert baseline["baseline_top10_precision"] == 0.825


def test_load_regression_top10_baseline_missing_target_returns_none(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["xgboost"],
            "target": ["race"],
            "mae": [2.4],
            "top10_precision": [0.825],
            "top10_recall": [0.825],
        }
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    assert load_regression_top10_baseline(summary_path, "qualifying") is None


def test_build_comparison_table_includes_both_targets(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["xgboost", "xgboost"],
            "target": ["race", "qualifying"],
            "mae": [2.4, 3.5],
            "top10_precision": [0.825, 0.725],
            "top10_recall": [0.825, 0.725],
        }
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    race_folds = pd.DataFrame(
        {
            "test_season": [2025],
            "precision": [0.7],
            "recall": [0.7],
            "log_loss": [0.5],
            "brier_score": [0.2],
        }
    )
    qual_folds = pd.DataFrame(
        {
            "test_season": [2025],
            "precision": [0.6],
            "recall": [0.6],
            "log_loss": [0.6],
            "brier_score": [0.25],
        }
    )

    comparison = build_comparison_table(race_folds, qual_folds, summary_path)
    assert set(comparison["classifier"]) == {"top10_race", "top10_qualifying"}
    race_row = comparison[comparison["classifier"] == "top10_race"].iloc[0]
    assert race_row["baseline_model"] == "xgboost"
    assert race_row["classifier_precision"] == 0.7
