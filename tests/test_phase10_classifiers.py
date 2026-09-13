import numpy as np
import pandas as pd
import pytest

from phase10_classifiers import (
    TOP10_QUALIFYING_DROP_COLS,
    TOP10_RACE_DROP_COLS,
    WINNER_PODIUM_DROP_COLS,
    _fit_calibrated_binary,
    train_top10_qualifying,
    train_top10_race,
    train_winner_podium,
)

# Columns that are either the race outcome re-encoded (dnf_flag) or measured
# during the race/qualifying session itself, so none of them are knowable at
# prediction time.
IN_RACE_LEAKAGE_COLS = [
    "dnf_flag",
    "race_air_temp",
    "race_track_temp",
    "race_humidity",
    "race_rainfall",
    "race_is_wet",
    "num_stints",
    "avg_tyre_life",
    "max_tyre_life",
    "avg_pit_time",
    "total_pit_time_lost",
    # race_track_temp minus the qualifying track temp: still carries a race
    # measurement, so it leaks into pre-race models too, not just pre-qualifying.
    "temp_delta_quali_race",
]


@pytest.mark.parametrize(
    "drop_cols",
    [WINNER_PODIUM_DROP_COLS, TOP10_RACE_DROP_COLS, TOP10_QUALIFYING_DROP_COLS],
    ids=["winner_podium", "top10_race", "top10_qualifying"],
)
def test_drop_cols_exclude_in_race_leakage_columns(drop_cols):
    missing = [col for col in IN_RACE_LEAKAGE_COLS if col not in drop_cols]
    assert missing == []


@pytest.mark.parametrize(
    "drop_cols",
    [WINNER_PODIUM_DROP_COLS, TOP10_RACE_DROP_COLS, TOP10_QUALIFYING_DROP_COLS],
    ids=["winner_podium", "top10_race", "top10_qualifying"],
)
def test_all_models_drop_race_session_temp_delta(drop_cols):
    # temp_delta_quali_race = race_track_temp - quali_track_temp, so it embeds a
    # race-session measurement. Unknowable before qualifying AND before the race.
    assert "temp_delta_quali_race" in drop_cols


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
    assert {"top10_cut_precision", "top10_cut_recall"}.issubset(fold_metrics.columns)
    assert (tmp_path / "fold_metrics.csv").exists()
    assert (tmp_path / "calibration_table_last_fold.csv").exists()
    predict_files = list(tmp_path.glob("predict_2025_round*.csv"))
    assert len(predict_files) == 1


def test_fit_calibrated_binary_all_zero_train_returns_near_zero_probs():
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 0])
    X_test = pd.DataFrame({"feat": [4.0, 5.0]})

    y_prob = _fit_calibrated_binary(X_train, y_train, X_test)

    assert isinstance(y_prob, np.ndarray)
    assert len(y_prob) == len(X_test)
    assert all(p == pytest.approx(0.01) for p in y_prob)


def test_fit_calibrated_binary_all_one_train_returns_near_one_probs():
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1, 1, 1])
    X_test = pd.DataFrame({"feat": [4.0, 5.0]})

    y_prob = _fit_calibrated_binary(X_train, y_train, X_test)

    assert isinstance(y_prob, np.ndarray)
    assert len(y_prob) == len(X_test)
    assert all(p == pytest.approx(0.99) for p in y_prob)


def _single_class_first_season_dataset() -> pd.DataFrame:
    """2023 has zero top-10 finishers, so the [2023]->2024 fold trains on one class."""
    rows = []
    for season in (2023, 2024, 2025):
        for round_ in (1, 2, 3):
            for driver_idx in range(12):
                # 2023 grid finishes 11th..22nd only: no positive labels at all.
                position = driver_idx + (11 if season == 2023 else 1)
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


def test_train_top10_race_survives_single_class_training_fold(tmp_path):
    """End-to-end guard: the degenerate single-class fallback must flow through
    the metrics call without raising and without an exploding log loss."""
    df = _single_class_first_season_dataset()

    fold_metrics = train_top10_race(df, output_dir=tmp_path)

    assert len(fold_metrics) == 2
    degenerate = fold_metrics[fold_metrics["test_season"] == 2024].iloc[0]
    assert np.isfinite(degenerate["log_loss"])
    # Hard 0.0/1.0 probabilities would push log_loss to ~18 here.
    assert degenerate["log_loss"] < 5.0
    assert 0.0 <= degenerate["brier_score"] <= 1.0
    assert 0.0 <= degenerate["top10_cut_precision"] <= 1.0


def test_train_top10_qualifying_writes_fold_metrics(tmp_path):
    df = _synthetic_top10_dataset()
    fold_metrics = train_top10_qualifying(df, output_dir=tmp_path)

    assert len(fold_metrics) == 2
    assert {"precision", "recall", "log_loss", "brier_score", "roc_auc"}.issubset(
        fold_metrics.columns
    )
    assert {"top10_cut_precision", "top10_cut_recall"}.issubset(fold_metrics.columns)
    assert (tmp_path / "fold_metrics.csv").exists()
    predict_files = list(tmp_path.glob("predict_2025_round*.csv"))
    assert len(predict_files) == 1


@pytest.mark.parametrize(
    "trainer",
    [train_winner_podium, train_top10_race, train_top10_qualifying],
    ids=["winner_podium", "top10_race", "top10_qualifying"],
)
def test_prediction_file_contains_only_the_round_it_is_named_after(trainer, tmp_path):
    # The synthetic top-10 dataset has 3 rounds x 12 drivers per season, so an
    # unfiltered dump would be 36 rows under a "round03" filename.
    df = _synthetic_top10_dataset()
    trainer(df, output_dir=tmp_path)

    predict_files = list(tmp_path.glob("predict_2025_round*.csv"))
    assert len(predict_files) == 1
    written = pd.read_csv(predict_files[0])
    assert predict_files[0].name == "predict_2025_round03.csv"
    assert set(written["round"]) == {3}
    assert len(written) == 12
