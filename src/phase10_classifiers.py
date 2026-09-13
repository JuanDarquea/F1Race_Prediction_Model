"""Phase 10: Winner/podium, top-10 race, and top-10 qualifying classifiers.

Unlike Phase 5's regression-then-rank-threshold approach, each of these is
trained directly on its true label (multi-class win/podium, or binary
top-10) and walk-forward validated across every available season.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

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
from common.walk_forward import expanding_season_folds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "feature_dataset.csv"
ELO_HISTORY_PATH = PROJECT_ROOT / "data" / "features" / "driver_elo_history.csv"
MODEL_DIR = PROJECT_ROOT / "models"

NON_FEATURE_COLS = ["season", "round", "driver_id"]

# Columns that are only known once the session has been run. `dnf_flag` is
# `status` numerically re-encoded (see phase4_feature_engineering), and the rest
# are weather/tyre/pit measurements taken during the race itself, so none of
# them can be fed to a pre-race prediction without leaking the outcome.
IN_RACE_MEASUREMENT_COLS = [
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
    # race_track_temp minus the qualifying track temp (phase4:254). Differencing
    # does not launder the leak: it still carries a race-session measurement, so
    # it is unknowable before qualifying AND before the race.
    "temp_delta_quali_race",
]

WINNER_PODIUM_DROP_COLS = [
    "race_points",
    "sprint_position",
    "sprint_points",
    "sprint_qualifying_position",
    "status",
    "driver_id",
] + IN_RACE_MEASUREMENT_COLS


def load_dataset_with_elo(
    features_path: Path = FEATURE_PATH, elo_path: Path = ELO_HISTORY_PATH
) -> pd.DataFrame:
    """Join feature_dataset.csv with Phase 9's Elo history on (season, round, driver_name)."""
    df = pd.read_csv(features_path)
    elo = pd.read_csv(elo_path)[["season", "round", "driver_name", "elo_pre_race"]]
    merged = df.merge(elo, on=["season", "round", "driver_name"], how="left")
    missing = int(merged["elo_pre_race"].isna().sum())
    if missing:
        print(f"[warn] {missing} rows had no Elo match; defaulting to 1500.0")
        merged["elo_pre_race"] = merged["elo_pre_race"].fillna(1500.0)
    return merged


def _finish_class(race_position: pd.Series) -> pd.Series:
    finish_class = pd.Series(0, index=race_position.index)
    finish_class[race_position == 3] = 3
    finish_class[race_position == 2] = 2
    finish_class[race_position == 1] = 1
    return finish_class


def _latest_season_round(predictions: pd.DataFrame) -> Tuple[int, int]:
    """The most recent (season, round) present in a fold's prediction frame."""
    latest_season = int(predictions["season"].max())
    latest_round = int(
        predictions.loc[predictions["season"] == latest_season, "round"].max()
    )
    return latest_season, latest_round


def _latest_prediction_filename(predictions: pd.DataFrame) -> str:
    latest_season, latest_round = _latest_season_round(predictions)
    return f"predict_{latest_season}_round{latest_round:02d}.csv"


def _write_latest_round_predictions(
    predictions: pd.DataFrame, output_dir: Path
) -> Path:
    """Write only the latest (season, round) slice of a fold's predictions.

    A fold's test set spans the whole test season, but the file is named after a
    single round, so it is filtered down to that round before writing - otherwise
    predict_<season>_round<NN>.csv holds every round of the season.
    """
    latest_season, latest_round = _latest_season_round(predictions)
    latest = predictions[
        (predictions["season"] == latest_season)
        & (predictions["round"] == latest_round)
    ]
    path = output_dir / _latest_prediction_filename(predictions)
    latest.to_csv(path, index=False)
    return path


def train_winner_podium(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Walk-forward train/evaluate the winner/podium multi-class classifier.

    Writes model fold metrics and the latest fold's predictions under
    output_dir; returns the fold-metrics DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    working = df.dropna(subset=["race_position"]).copy()
    working["finish_class"] = _finish_class(working["race_position"])

    fold_rows = []
    last_fold_predictions = None
    for train_seasons, test_season in expanding_season_folds(working):
        train_raw = working[working["season"].isin(train_seasons)]
        test_raw = working[working["season"] == test_season]

        train_enc = clean_and_encode(
            train_raw, drop_cols=WINNER_PODIUM_DROP_COLS + ["race_position"]
        )
        test_enc = clean_and_encode(
            test_raw, drop_cols=WINNER_PODIUM_DROP_COLS + ["race_position"]
        )
        train_enc, test_enc = align_train_test(train_enc, test_enc)

        cols = feature_columns(train_enc, NON_FEATURE_COLS + ["finish_class"])
        X_train, y_train = train_enc[cols], train_enc["finish_class"]
        X_test, y_test = test_enc[cols], test_enc["finish_class"]

        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
        )
        model.fit(X_train, y_train)
        classes = list(model.classes_)
        proba = model.predict_proba(X_test)

        metrics = multiclass_top3_metrics(y_test.to_numpy(), proba, classes)
        win_idx = classes.index(1) if 1 in classes else None
        if win_idx is not None:
            metrics["top3_accuracy"] = top3_accuracy_by_race(
                test_raw.reset_index(drop=True)[["season", "round"]],
                y_test.to_numpy(),
                proba[:, win_idx],
            )
        else:
            metrics["top3_accuracy"] = float("nan")
        metrics["train_seasons"] = ",".join(str(s) for s in train_seasons)
        metrics["test_season"] = test_season
        fold_rows.append(metrics)

        top3_idx = [classes.index(c) for c in (1, 2, 3) if c in classes]
        predictions = test_raw[["driver_name", "team", "season", "round"]].reset_index(
            drop=True
        )
        predictions["p_win"] = proba[:, win_idx] if win_idx is not None else 0.0
        predictions["p_top3"] = proba[:, top3_idx].sum(axis=1)
        last_fold_predictions = predictions

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    if last_fold_predictions is not None:
        _write_latest_round_predictions(last_fold_predictions, output_dir)
    return fold_metrics


TOP10_RACE_DROP_COLS = [
    "race_points",
    "sprint_position",
    "sprint_points",
    "sprint_qualifying_position",
    "status",
    "driver_id",
] + IN_RACE_MEASUREMENT_COLS


def _fit_calibrated_binary(X_train, y_train, X_test):
    if y_train.nunique() < 2:
        # Degenerate fold: nothing to learn, so lean towards the only class seen
        # but stay short of certainty — hard 0.0/1.0 sends log_loss to ~18 the
        # moment a single test label disagrees. Returns an ndarray like every
        # other path, so the downstream metrics helpers work unchanged.
        only_class = int(y_train.iloc[0])
        clipped = 0.99 if only_class == 1 else 0.01
        return np.full(len(X_test), clipped)
    base_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated.predict_proba(X_test)[:, 1]


def train_top10_race(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Walk-forward train/evaluate a binary top-10-race-finish classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    working = df.dropna(subset=["race_position"]).copy()
    working["is_top10"] = (working["race_position"] <= 10).astype(int)

    fold_rows = []
    last_fold_predictions = None
    last_y_test = None
    last_y_prob = None
    for train_seasons, test_season in expanding_season_folds(working):
        train_raw = working[working["season"].isin(train_seasons)]
        test_raw = working[working["season"] == test_season]

        train_enc = clean_and_encode(
            train_raw, drop_cols=TOP10_RACE_DROP_COLS + ["race_position"]
        )
        test_enc = clean_and_encode(
            test_raw, drop_cols=TOP10_RACE_DROP_COLS + ["race_position"]
        )
        train_enc, test_enc = align_train_test(train_enc, test_enc)

        cols = feature_columns(train_enc, NON_FEATURE_COLS + ["is_top10"])
        X_train, y_train = train_enc[cols], train_enc["is_top10"]
        X_test, y_test = test_enc[cols], test_enc["is_top10"]

        y_prob = _fit_calibrated_binary(X_train, y_train, X_test)

        metrics = binary_classification_metrics(y_test.to_numpy(), y_prob)
        metrics.update(
            top10_cut_precision_recall(
                test_raw.reset_index(drop=True)[["season", "round"]],
                y_test.to_numpy(),
                y_prob,
            )
        )
        metrics["train_seasons"] = ",".join(str(s) for s in train_seasons)
        metrics["test_season"] = test_season
        fold_rows.append(metrics)

        predictions = test_raw[["driver_name", "team", "season", "round"]].reset_index(
            drop=True
        )
        predictions["p_top10"] = y_prob
        last_fold_predictions = predictions
        last_y_test, last_y_prob = y_test.to_numpy(), y_prob

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    if last_fold_predictions is not None:
        _write_latest_round_predictions(last_fold_predictions, output_dir)
        calibration_table(last_y_test, last_y_prob).to_csv(
            output_dir / "calibration_table_last_fold.csv", index=False
        )
    return fold_metrics


TOP10_QUALIFYING_DROP_COLS = [
    "race_position",
    "race_points",
    "sprint_position",
    "sprint_points",
    "sprint_qualifying_position",
    "status",
    "driver_id",
] + IN_RACE_MEASUREMENT_COLS


def train_top10_qualifying(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Walk-forward train/evaluate a binary top-10-qualifying classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    working = df.dropna(subset=["qualifying_position"]).copy()
    working["is_top10"] = (working["qualifying_position"] <= 10).astype(int)

    fold_rows = []
    last_fold_predictions = None
    last_y_test = None
    last_y_prob = None
    for train_seasons, test_season in expanding_season_folds(working):
        train_raw = working[working["season"].isin(train_seasons)]
        test_raw = working[working["season"] == test_season]

        train_enc = clean_and_encode(
            train_raw, drop_cols=TOP10_QUALIFYING_DROP_COLS + ["qualifying_position"]
        )
        test_enc = clean_and_encode(
            test_raw, drop_cols=TOP10_QUALIFYING_DROP_COLS + ["qualifying_position"]
        )
        train_enc, test_enc = align_train_test(train_enc, test_enc)

        cols = feature_columns(train_enc, NON_FEATURE_COLS + ["is_top10"])
        X_train, y_train = train_enc[cols], train_enc["is_top10"]
        X_test, y_test = test_enc[cols], test_enc["is_top10"]

        y_prob = _fit_calibrated_binary(X_train, y_train, X_test)

        metrics = binary_classification_metrics(y_test.to_numpy(), y_prob)
        metrics.update(
            top10_cut_precision_recall(
                test_raw.reset_index(drop=True)[["season", "round"]],
                y_test.to_numpy(),
                y_prob,
            )
        )
        metrics["train_seasons"] = ",".join(str(s) for s in train_seasons)
        metrics["test_season"] = test_season
        fold_rows.append(metrics)

        predictions = test_raw[["driver_name", "team", "season", "round"]].reset_index(
            drop=True
        )
        predictions["p_top10_qualifying"] = y_prob
        last_fold_predictions = predictions
        last_y_test, last_y_prob = y_test.to_numpy(), y_prob

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    if last_fold_predictions is not None:
        _write_latest_round_predictions(last_fold_predictions, output_dir)
        calibration_table(last_y_test, last_y_prob).to_csv(
            output_dir / "calibration_table_last_fold.csv", index=False
        )
    return fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10: Winner/top-10 classifiers")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--elo-history", default=str(ELO_HISTORY_PATH))
    args = parser.parse_args()

    df = load_dataset_with_elo(Path(args.features), Path(args.elo_history))

    winner_metrics = train_winner_podium(df, MODEL_DIR / "winner_podium")
    print("[winner_podium] fold metrics:\n", winner_metrics)

    top10_race_metrics = train_top10_race(df, MODEL_DIR / "top10_race")
    print("[top10_race] fold metrics:\n", top10_race_metrics)

    top10_qual_metrics = train_top10_qualifying(df, MODEL_DIR / "top10_qualifying")
    print("[top10_qualifying] fold metrics:\n", top10_qual_metrics)


if __name__ == "__main__":
    main()
