from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import fastf1
from xgboost import XGBRegressor

from phase1_data_collection import collect_for_seasons
from phase4_feature_engineering import build_feature_dataset
from phase5_model_training import IDENTIFIER_COLS, TARGET_CONFIGS, _prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TRACK_TYPE_PATH = PROJECT_ROOT / "data" / "track_types.csv"


def _available_models() -> List[str]:
    candidates = ["xgboost", "random_forest", "linear_regression"]
    return [name for name in candidates if (MODEL_DIR / name).exists()]


def _load_model(model_name: str, suffix: str):
    model_dir = MODEL_DIR / model_name
    if model_name == "xgboost":
        model_path = model_dir / f"model{suffix}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        model = XGBRegressor()
        model.load_model(model_path)
        return model

    model_path = model_dir / f"model{suffix}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return pd.read_pickle(model_path)


def _select_weekend(
    df: pd.DataFrame, year: int, round_number: int | None
) -> tuple[int, str]:
    weekends = (
        df[df["season"] == year][["round", "track"]]
        .drop_duplicates()
        .sort_values("round")
        .reset_index(drop=True)
    )

    if round_number is not None:
        if not weekends.empty:
            match = weekends[weekends["round"] == round_number]
            if not match.empty:
                track = match.iloc[0]["track"]
                return int(round_number), str(track)

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        schedule_match = schedule[schedule["RoundNumber"] == round_number]
        if schedule_match.empty:
            raise ValueError(f"Round {round_number} not found for {year}.")
        return int(round_number), str(schedule_match.iloc[0]["EventName"])

    if weekends.empty:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            raise ValueError(f"No schedule found for season {year}.")
        last = schedule.iloc[-1]
        return int(last["RoundNumber"]), str(last["EventName"])

    # Default to latest round in data if not provided
    last = weekends.iloc[-1]
    return int(last["round"]), str(last["track"])


def _prepare_training_columns(df: pd.DataFrame, target_key: str) -> List[str]:
    cfg = TARGET_CONFIGS[target_key]
    train_raw = df[df["season"].isin([2023, 2024])]
    test_raw = df[df["season"] == 2025]

    if train_raw.empty or test_raw.empty:
        raise ValueError("Not enough data to build training columns (need 2023-2025).")

    train = _prepare_features(
        train_raw, cfg.target_col, cfg.required_cols, cfg.drop_cols
    )
    test = _prepare_features(test_raw, cfg.target_col, cfg.required_cols, cfg.drop_cols)

    train, test = train.align(test, join="left", axis=1, fill_value=0)

    feature_cols = [
        col
        for col in train.columns
        if col not in IDENTIFIER_COLS and col != cfg.target_col
    ]
    return feature_cols


def _align_to_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = 0
    extra_cols = [col for col in aligned.columns if col not in feature_cols]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)
    return aligned[feature_cols].fillna(0)


def _prepare_features_inference(df: pd.DataFrame, target_key: str) -> pd.DataFrame:
    cfg = TARGET_CONFIGS[target_key]
    prepared = df.copy()

    # Ensure required columns exist and are non-null to avoid dropping rows.
    if cfg.target_col not in prepared.columns:
        prepared[cfg.target_col] = 0
    # Never use current-year target values as features in inference.
    prepared[cfg.target_col] = 0

    for col in cfg.required_cols:
        if col not in prepared.columns:
            prepared[col] = 0
        prepared[col] = prepared[col].fillna(0)

    # Pre-fill numeric columns to avoid median warnings on all-NaN columns.
    numeric_cols = prepared.select_dtypes(include=["number"]).columns
    prepared[numeric_cols] = prepared[numeric_cols].fillna(0)

    return _prepare_features(prepared, cfg.target_col, cfg.required_cols, cfg.drop_cols)


def _predict_ensemble(
    models: List[str],
    features_raw: pd.DataFrame,
    target_key: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    cfg = TARGET_CONFIGS[target_key]
    features = _prepare_features_inference(features_raw, target_key)
    if features.empty:
        raise ValueError(f"Not enough data to prepare {target_key} features.")

    X = _align_to_columns(features, feature_cols)

    preds_per_model = []
    ranks_per_model = []
    for model_name in models:
        model = _load_model(model_name, cfg.output_suffix)
        preds = np.nan_to_num(model.predict(X))
        preds_per_model.append(preds)
        ranks = pd.Series(preds).rank(method="first").to_numpy()
        ranks_per_model.append(ranks)

    avg_rank = np.mean(np.vstack(ranks_per_model), axis=0)
    avg_pred = np.mean(np.vstack(preds_per_model), axis=0)

    # If any NaNs sneak in, fall back to ranking by avg_pred.
    if np.isnan(avg_rank).any():
        avg_rank = pd.Series(avg_pred).rank(method="first").to_numpy()

    out = features_raw.loc[
        features.index, ["driver_name", "team", "season", "round", "track"]
    ].copy()
    out = out.rename(columns={"track": "grand_prix_name"})
    out["avg_predicted_position"] = avg_pred
    out["avg_rank"] = avg_rank
    out["predicted_rank"] = (
        pd.Series(avg_rank, index=out.index).rank(method="first").astype(int)
    )

    return out.sort_values("predicted_rank")


def _pole_probability(
    models: List[str],
    features_raw: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    cfg = TARGET_CONFIGS["qualifying"]
    features = _prepare_features_inference(features_raw, "qualifying")
    if features.empty:
        raise ValueError("Not enough data for qualifying prediction.")

    X = _align_to_columns(features, feature_cols)

    preds_per_model = []
    for model_name in models:
        model = _load_model(model_name, cfg.output_suffix)
        preds = np.nan_to_num(model.predict(X))
        preds_per_model.append(preds)

    avg_pred = np.mean(np.vstack(preds_per_model), axis=0)
    scores = np.exp(-avg_pred)
    pole_prob = scores / scores.sum()
    return pd.Series(pole_prob, index=features.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Predict 2026 race weekend")
    parser.add_argument("--year", type=int, default=2026, help="Season year")
    parser.add_argument("--round", type=int, help="Round number to predict")
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"],
        help="Sessions to refresh for the selected weekend",
    )
    args = parser.parse_args()
    warnings: List[str] = []

    models = _available_models()
    if not models:
        raise ValueError("No trained models found. Run Phase 5 training first.")

    # Rebuild feature dataset with latest raw data
    full_features = build_feature_dataset(TRACK_TYPE_PATH)

    if args.round is None:
        season_rows = full_features[full_features["season"] == args.year]
        if season_rows.empty:
            print(f"No {args.year} data found yet; defaulting to Round 1.")
            round_to_fetch = 1
        else:
            round_to_fetch, _ = _select_weekend(full_features, args.year, None)
            print(f"Auto-selected latest round: {round_to_fetch}")
    else:
        round_to_fetch = args.round

    has_round_data = not full_features[
        (full_features["season"] == args.year)
        & (full_features["round"] == round_to_fetch)
    ].empty

    if has_round_data:
        try:
            collect_for_seasons(
                years=[args.year], session_codes=args.sessions, rounds=[round_to_fetch]
            )
        except Exception as exc:
            warnings.append(f"Download error: {exc}")
            print(f"Download error: {exc}")
    else:
        msg = (
            f"No session data found for {args.year} round {round_to_fetch}; "
            "skipping download and using historical features only."
        )
        warnings.append(msg)
        print(msg)

    # Rebuild feature dataset with latest raw data
    full_features = build_feature_dataset(TRACK_TYPE_PATH)

    selected_round, selected_track = _select_weekend(
        full_features, args.year, round_to_fetch
    )
    weekend_features = full_features[
        (full_features["season"] == args.year)
        & (full_features["round"] == selected_round)
    ]

    if weekend_features.empty:
        latest_season = int(full_features["season"].max())
        latest_round = int(
            full_features[full_features["season"] == latest_season]["round"].max()
        )
        weekend_features = full_features[
            (full_features["season"] == latest_season)
            & (full_features["round"] == latest_round)
        ]
        msg = (
            f"No feature rows found for {args.year} round {selected_round}. "
            f"Using historical proxy: season {latest_season}, round {latest_round}."
        )
        warnings.append(msg)
        print(msg)

        # Override to requested race metadata for output.
        weekend_features = weekend_features.copy()
        weekend_features["season"] = args.year
        weekend_features["round"] = selected_round
        weekend_features["track"] = selected_track

    race_feature_cols = _prepare_training_columns(full_features, "race")
    race_preds = _predict_ensemble(models, weekend_features, "race", race_feature_cols)

    practice_available = bool(
        (weekend_features.get("practice_pace") is not None)
        and weekend_features["practice_pace"].notna().any()
    )
    if not practice_available:
        warnings.append("Practice data not available for this weekend.")

    try:
        qual_feature_cols = _prepare_training_columns(full_features, "qualifying")
        pole_pct = _pole_probability(models, weekend_features, qual_feature_cols)
        race_preds["pole_pct"] = pole_pct.values * 100
    except FileNotFoundError:
        warnings.append(
            "Qualifying models not found; run Phase 5 training to enable Pole %."
        )
        ranks = race_preds["predicted_rank"].to_numpy(dtype=float)
        if np.isnan(ranks).any():
            ranks = (
                pd.Series(race_preds["avg_predicted_position"])
                .rank(method="first")
                .to_numpy()
            )
        scores = np.exp(-ranks)
        race_preds["pole_pct"] = (scores / scores.sum()) * 100
    except Exception:
        warnings.append(
            "Pole % could not be computed from qualifying; using race-rank proxy."
        )
        # Fallback: convert race predicted ranks into a softmax-like pole probability
        ranks = race_preds["predicted_rank"].to_numpy(dtype=float)
        if np.isnan(ranks).any():
            ranks = (
                pd.Series(race_preds["avg_predicted_position"])
                .rank(method="first")
                .to_numpy()
            )
        scores = np.exp(-ranks)
        pole_pct = (scores / scores.sum()) * 100
        race_preds["pole_pct"] = pole_pct

    top10 = race_preds.head(10).copy()
    output = top10[["driver_name", "pole_pct", "predicted_rank"]]
    output = output.rename(
        columns={
            "driver_name": "Driver",
            "pole_pct": "Pole %",
            "predicted_rank": "Predicted Finish Position",
        }
    )

    output_path = (
        MODEL_DIR / "ensemble" / f"predict_{args.year}_round{selected_round:02d}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    report_path = (
        MODEL_DIR / "ensemble" / f"report_{args.year}_round{selected_round:02d}.txt"
    )
    with report_path.open("w") as f:
        f.write(f"Phase 6 Prediction Report - {args.year} Round {selected_round}\n")
        f.write("============================================\n\n")
        f.write(f"Grand Prix: {selected_track}\n")
        f.write(f"Practice data available: {practice_available}\n")
        if warnings:
            f.write("Warnings / Potential Underperformance Reasons:\n")
            for item in warnings:
                f.write(f"- {item}\n")
        else:
            f.write("No warnings detected.\n")

    print(f"Predictions for {selected_track} (Round {selected_round})")
    print(output)
    print(f"Saved -> {output_path}")
    print(f"Report -> {report_path}")

    top3 = top10.sort_values("pole_pct", ascending=False).head(3)
    print("\nPole candidates (Top 3):")
    print(top3[["driver_name", "pole_pct", "predicted_rank"]])


if __name__ == "__main__":
    main()
