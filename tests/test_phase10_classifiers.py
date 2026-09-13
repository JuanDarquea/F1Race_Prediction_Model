import pandas as pd

from phase10_classifiers import (
    _fit_calibrated_binary,
    train_top10_race,
    train_winner_podium,
)


def _synthetic_dataset() -> pd.DataFrame:
    rows = []
    for season in (2023, 2024, 2025):
        for round_ in (1, 2):
            for driver_idx, driver in enumerate(["A", "B", "C", "D"]):
                rows.append(
                    {
                        "season": season,
                        "round": round_,
                        "driver_name": driver,
                        "driver_id": driver_idx + 1,
                        "team": "Team" + driver,
                        "track": "Track1",
                        "status": "Finished",
                        "race_position": driver_idx + 1,
                        "race_points": 0,
                        "sprint_position": None,
                        "sprint_points": None,
                        "sprint_qualifying_position": None,
                        "elo_pre_race": 1500.0 + driver_idx * 10,
                        "practice_pace": 90.0 + driver_idx,
                    }
                )
    return pd.DataFrame(rows)


def test_train_winner_podium_writes_fold_metrics_and_predictions(tmp_path):
    df = _synthetic_dataset()
    fold_metrics = train_winner_podium(df, output_dir=tmp_path)

    assert len(fold_metrics) == 2  # folds: [2023]->2024, [2023,2024]->2025
    assert {"win_log_loss", "top3_accuracy", "test_season"}.issubset(
        fold_metrics.columns
    )
    assert (tmp_path / "fold_metrics.csv").exists()
    predict_files = list(tmp_path.glob("predict_2025_round*.csv"))
    assert len(predict_files) == 1


def _synthetic_top10_dataset() -> pd.DataFrame:
    rows = []
    for season in (2023, 2024, 2025):
        for round_ in (1, 2, 3):
            for driver_idx in range(12):
                position = driver_idx + 1
                rows.append(
                    {
                        "season": season,
                        "round": round_,
                        "driver_name": f"D{driver_idx}",
                        "driver_id": driver_idx + 1,
                        "team": f"Team{driver_idx % 4}",
                        "track": "Track1",
                        "status": "Finished",
                        "race_position": position,
                        "qualifying_position": position,
                        "race_points": 0,
                        "sprint_position": None,
                        "sprint_points": None,
                        "sprint_qualifying_position": None,
                        "elo_pre_race": 1500.0 + driver_idx * 5,
                        "practice_pace": 90.0 + driver_idx,
                    }
                )
    return pd.DataFrame(rows)


def test_train_top10_race_writes_fold_metrics_and_calibration(tmp_path):
    df = _synthetic_top10_dataset()
    fold_metrics = train_top10_race(df, output_dir=tmp_path)

    assert len(fold_metrics) == 2
    assert {"precision", "recall", "log_loss", "brier_score", "roc_auc"}.issubset(
        fold_metrics.columns
    )
    assert (tmp_path / "fold_metrics.csv").exists()
    assert (tmp_path / "calibration_table_last_fold.csv").exists()
    predict_files = list(tmp_path.glob("predict_2025_round*.csv"))
    assert len(predict_files) == 1


def test_fit_calibrated_binary_all_zero_train_returns_near_zero_probs():
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 0])
    X_test = pd.DataFrame({"feat": [4.0, 5.0]})

    y_prob = _fit_calibrated_binary(X_train, y_train, X_test)

    assert len(y_prob) == len(X_test)
    assert all(p == 0.0 for p in y_prob)


def test_fit_calibrated_binary_all_one_train_returns_near_one_probs():
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1, 1, 1])
    X_test = pd.DataFrame({"feat": [4.0, 5.0]})

    y_prob = _fit_calibrated_binary(X_train, y_train, X_test)

    assert len(y_prob) == len(X_test)
    assert all(p == 1.0 for p in y_prob)
