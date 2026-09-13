"""Phase 10: Winner/podium, top-10 race, and top-10 qualifying classifiers.

Unlike Phase 5's regression-then-rank-threshold approach, each of these is
trained directly on its true label (multi-class win/podium, or binary
top-10) and walk-forward validated across every available season.
"""

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from common.classifier_utils import (
    align_train_test,
    clean_and_encode,
    feature_columns,
    multiclass_top3_metrics,
    top3_accuracy_by_race,
)
from common.walk_forward import expanding_season_folds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "feature_dataset.csv"
ELO_HISTORY_PATH = PROJECT_ROOT / "data" / "features" / "driver_elo_history.csv"
MODEL_DIR = PROJECT_ROOT / "models"

NON_FEATURE_COLS = ["season", "round", "driver_id"]

WINNER_PODIUM_DROP_COLS = [
    "race_points",
    "sprint_position",
    "sprint_points",
    "sprint_qualifying_position",
    "status",
    "driver_id",
]


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


def _latest_prediction_filename(predictions: pd.DataFrame) -> str:
    latest_season = int(predictions["season"].max())
    latest_round = int(
        predictions[predictions["season"] == latest_season]["round"].max()
    )
    return f"predict_{latest_season}_round{latest_round:02d}.csv"


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
        last_fold_predictions.to_csv(
            output_dir / _latest_prediction_filename(last_fold_predictions), index=False
        )
    return fold_metrics
