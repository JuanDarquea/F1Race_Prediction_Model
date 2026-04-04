import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "feature_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"


IDENTIFIER_COLS = [
    "driver_name",
    "driver_id",
    "team",
    "season",
    "round",
    "track",
]
TARGET_COL = "race_position"


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()

    # Drop rows without target or key weekend signal.
    feature_df = feature_df.dropna(subset=[TARGET_COL, "qualifying_position"])
    # Drop non-feature columns that can leak or are non-numeric.
    if "status" in feature_df.columns:
        feature_df = feature_df.drop(columns=["status"])

    # Fill numeric NaNs with median (simple baseline strategy).
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if col == TARGET_COL:
            continue
        median_value = feature_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        feature_df[col] = feature_df[col].fillna(median_value)

    # One-hot encode categorical features.
    categorical_cols = ["driver_name", "team", "track_type", "track"]
    categorical_cols = [col for col in categorical_cols if col in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)

    # Ensure no object columns remain (XGBoost requires numeric/bool).
    object_cols = feature_df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        feature_df = feature_df.drop(columns=list(object_cols))

    return feature_df


def _split_data(df: pd.DataFrame):
    train = df[df["season"].isin([2023, 2024])].copy()
    test = df[df["season"] == 2025].copy()
    return train, test


def _top10_metrics(pred_df: pd.DataFrame) -> dict:
    per_race = pred_df.groupby(["season", "round"], dropna=True)
    precisions = []
    recalls = []

    for _, group in per_race:
        true_top10 = set(group.nsmallest(10, "race_position").index)
        pred_top10 = set(group.nsmallest(10, "predicted_position").index)

        tp = len(true_top10 & pred_top10)
        precision = tp / 10 if group.shape[0] >= 10 else tp / max(group.shape[0], 1)
        recall = tp / 10 if group.shape[0] >= 10 else tp / max(group.shape[0], 1)
        precisions.append(precision)
        recalls.append(recall)

    return {
        "top10_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "top10_recall": float(np.mean(recalls)) if recalls else float("nan"),
    }


def _spearman_corr(pred_df: pd.DataFrame) -> float:
    return pred_df["race_position"].corr(
        pred_df["predicted_position"], method="spearman"
    )


def _predict_with_ranking(
    model: XGBRegressor,
    test_raw: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    preds = model.predict(X_test)
    out = test_raw.loc[X_test.index].copy()
    out["predicted_position"] = preds

    # Rank within each race
    out["predicted_rank"] = (
        out.groupby(["season", "round"], dropna=True)["predicted_position"]
        .rank(method="first")
        .astype(int)
    )
    return out


def _write_report(report_path: Path, metrics: dict, feature_count: int) -> None:
    with report_path.open("w") as f:
        f.write("Phase 5 - First ML Model Report\n")
        f.write("================================\n\n")
        f.write("Model: XGBoost Regressor (predict race_position)\n")
        f.write("Train seasons: 2023, 2024\n")
        f.write("Test season: 2025\n\n")
        f.write(f"Features used: {feature_count}\n\n")
        f.write("Evaluation Metrics (2025):\n")
        f.write(f"- MAE: {metrics['mae']:.3f}\n")
        f.write(f"- Spearman: {metrics['spearman']:.3f}\n")
        f.write(f"- Top-10 Precision: {metrics['top10_precision']:.3f}\n")
        f.write(f"- Top-10 Recall: {metrics['top10_recall']:.3f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Train first ML model")
    parser.add_argument(
        "--features",
        default=str(FEATURE_PATH),
        help="Path to feature dataset",
    )
    parser.add_argument(
        "--report",
        default=str(MODEL_DIR / "phase5_report.txt"),
        help="Path to save model report",
    )
    parser.add_argument(
        "--predictions",
        default=str(MODEL_DIR / "phase5_predictions_2025.csv"),
        help="Path to save 2025 predictions",
    )
    parser.add_argument(
        "--model-path",
        default=str(MODEL_DIR / "phase5_xgboost.json"),
        help="Path to save trained model",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    train_raw, test_raw = _split_data(df)

    if train_raw.empty or test_raw.empty:
        raise ValueError("Training or testing split is empty. Check your seasons.")

    train = _prepare_features(train_raw)
    test = _prepare_features(test_raw)

    # Align columns between train and test
    train, test = train.align(test, join="left", axis=1, fill_value=0)

    # Remove identifiers from feature set
    feature_cols = [
        col for col in train.columns if col not in IDENTIFIER_COLS and col != TARGET_COL
    ]

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_test = test[feature_cols]
    y_test = test[TARGET_COL]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))

    ranked = _predict_with_ranking(model, test_raw, X_test)

    metrics = {
        "mae": mae,
        "spearman": _spearman_corr(ranked),
    }
    metrics.update(_top10_metrics(ranked))

    # Save predictions
    ranked = ranked.rename(columns={"track": "grand_prix_name"})
    output_cols = [
        "driver_name",
        "team",
        "season",
        "round",
        "grand_prix_name",
        "race_position",
        "predicted_position",
        "predicted_rank",
    ]
    ranked[output_cols].to_csv(args.predictions, index=False)

    # Save model
    model.save_model(args.model_path)

    # Write report
    _write_report(Path(args.report), metrics, len(feature_cols))

    print(f"Saved model -> {args.model_path}")
    print(f"Saved predictions -> {args.predictions}")
    print(f"Saved report -> {args.report}")


if __name__ == "__main__":
    main()
