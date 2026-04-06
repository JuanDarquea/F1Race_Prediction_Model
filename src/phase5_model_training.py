import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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


def _top10_confusion(pred_df: pd.DataFrame) -> dict:
    per_race = pred_df.groupby(["season", "round"], dropna=True)
    tp_total = fp_total = fn_total = tn_total = 0

    for _, group in per_race:
        if group.empty:
            continue
        true_top10 = set(group.nsmallest(10, "race_position").index)
        pred_top10 = set(group.nsmallest(10, "predicted_position").index)

        for idx in group.index:
            is_true = idx in true_top10
            is_pred = idx in pred_top10
            if is_true and is_pred:
                tp_total += 1
            elif is_pred and not is_true:
                fp_total += 1
            elif is_true and not is_pred:
                fn_total += 1
            else:
                tn_total += 1

    total = tp_total + fp_total + fn_total + tn_total
    accuracy = (tp_total + tn_total) / total if total else float("nan")
    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "tn": tn_total,
        "top10_accuracy": accuracy,
    }


def _top10_confusion_by_race(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    per_race = pred_df.groupby(["season", "round"], dropna=True)
    for (season, rnd), group in per_race:
        if group.empty:
            continue
        true_top10 = set(group.nsmallest(10, "race_position").index)
        pred_top10 = set(group.nsmallest(10, "predicted_position").index)
        tp = fp = fn = tn = 0
        for idx in group.index:
            is_true = idx in true_top10
            is_pred = idx in pred_top10
            if is_true and is_pred:
                tp += 1
            elif is_pred and not is_true:
                fp += 1
            elif is_true and not is_pred:
                fn += 1
            else:
                tn += 1
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else float("nan")
        grand_prix_name = None
        if "grand_prix_name" in group.columns:
            grand_prix_name = group["grand_prix_name"].iloc[0]
        elif "track" in group.columns:
            grand_prix_name = group["track"].iloc[0]

        rows.append(
            {
                "season": season,
                "round": rnd,
                "grand_prix_name": grand_prix_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "top10_accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def _plot_confusion_matrix(confusion: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.array(
        [[confusion["tp"], confusion["fp"]], [confusion["fn"], confusion["tn"]]]
    )
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Top-10 Confusion Matrix (Aggregated)")
    plt.xticks([0, 1], ["Pred Top10", "Pred Not"])
    plt.yticks([0, 1], ["Actual Top10", "Actual Not"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _spearman_corr(pred_df: pd.DataFrame) -> float:
    return pred_df["race_position"].corr(
        pred_df["predicted_position"], method="spearman"
    )


def _predict_with_ranking(
    model,
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


def _write_report(
    report_path: Path,
    model_name: str,
    metrics: dict,
    feature_count: int,
    confusion: dict | None,
) -> None:
    with report_path.open("w") as f:
        f.write("Phase 5 - First ML Model Report\n")
        f.write("================================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write("Train seasons: 2023, 2024\n")
        f.write("Test season: 2025\n\n")
        f.write(f"Features used: {feature_count}\n\n")
        f.write("Evaluation Metrics (2025):\n")
        f.write(f"- MAE: {metrics['mae']:.3f}\n")
        f.write(f"- Spearman: {metrics['spearman']:.3f}\n")
        f.write(f"- Top-10 Precision: {metrics['top10_precision']:.3f}\n")
        f.write(f"- Top-10 Recall: {metrics['top10_recall']:.3f}\n")
        if confusion:
            f.write(f"- Top-10 Accuracy: {confusion['top10_accuracy']:.3f}\n")
            f.write("\nTop-10 Confusion Matrix (Aggregated):\n")
            f.write(f"TP: {confusion['tp']}\n")
            f.write(f"FP: {confusion['fp']}\n")
            f.write(f"FN: {confusion['fn']}\n")
            f.write(f"TN: {confusion['tn']}\n")


def _save_sklearn_model(model, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(model, f)


def _save_sklearn_model_json(model, feature_cols: List[str], path: Path) -> None:
    payload = {"model": model.__class__.__name__}
    if hasattr(model, "coef_"):
        payload["coefficients"] = dict(zip(feature_cols, model.coef_.tolist()))
        payload["intercept"] = float(model.intercept_)
    if hasattr(model, "feature_importances_"):
        payload["feature_importances"] = dict(
            zip(feature_cols, model.feature_importances_.tolist())
        )
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _train_and_evaluate(
    model_name: str,
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> None:
    model_dir = MODEL_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_test = test[feature_cols]
    y_test = test[TARGET_COL]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))

    ranked = _predict_with_ranking(model, test_raw, X_test)
    ranked = ranked.rename(columns={"track": "grand_prix_name"})

    metrics = {
        "mae": mae,
        "spearman": _spearman_corr(ranked),
    }
    metrics.update(_top10_metrics(ranked))
    confusion = _top10_confusion(ranked) if args.top10_classification else None

    predictions_path = model_dir / "predictions_2025.csv"
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
    ranked[output_cols].to_csv(predictions_path, index=False)

    if isinstance(model, XGBRegressor):
        model_path = model_dir / "model.json"
        model.save_model(model_path)
    else:
        model_path = model_dir / "model.pkl"
        _save_sklearn_model(model, model_path)
        _save_sklearn_model_json(model, feature_cols, model_dir / "model.json")

    report_path = model_dir / "report.txt"
    _write_report(report_path, model_name, metrics, len(feature_cols), confusion)

    if args.top10_classification:
        class_out = ranked[
            [
                "driver_name",
                "team",
                "season",
                "round",
                "race_position",
                "predicted_rank",
            ]
        ].copy()
        class_out = class_out.rename(columns={"predicted_rank": "predicted_top10_rank"})
        class_out["is_top10_true"] = class_out["race_position"] <= 10
        class_out["is_top10_pred"] = class_out["predicted_top10_rank"] <= 10
        class_out.to_csv(model_dir / "top10_classification_2025.csv", index=False)
        per_race = _top10_confusion_by_race(ranked)
        per_race.to_csv(model_dir / "top10_confusion_by_race_2025.csv", index=False)
        if confusion:
            _plot_confusion_matrix(confusion, model_dir / "top10_confusion_matrix.png")

    print(f"[{model_name}] Saved model -> {model_path}")
    print(f"[{model_name}] Saved predictions -> {predictions_path}")
    print(f"[{model_name}] Saved report -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Train first ML model")
    parser.add_argument(
        "--features",
        default=str(FEATURE_PATH),
        help="Path to feature dataset",
    )
    parser.add_argument(
        "--top10-classification",
        action="store_true",
        help="Write top-10 classification output and confusion matrix.",
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

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    lr = LinearRegression()

    _train_and_evaluate("xgboost", xgb, train, test, test_raw, feature_cols, args)
    _train_and_evaluate("random_forest", rf, train, test, test_raw, feature_cols, args)
    _train_and_evaluate(
        "linear_regression", lr, train, test, test_raw, feature_cols, args
    )


if __name__ == "__main__":
    main()
