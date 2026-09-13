import pandas as pd

from phase10_classifiers import train_winner_podium


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
