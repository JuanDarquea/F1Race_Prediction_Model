import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path("/tmp/matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def _find_prediction_files(models_dir: Path) -> list[Path]:
    files: list[Path] = []
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for path in model_dir.glob("predictions_2025*.csv"):
            files.append(path)
    return sorted(files)


def _infer_target(df: pd.DataFrame) -> tuple[str, str]:
    if "race_position" in df.columns:
        return "race", "race_position"
    if "sprint_position" in df.columns:
        return "sprint", "sprint_position"
    if "qualifying_position" in df.columns:
        return "qualifying", "qualifying_position"
    raise ValueError("Could not infer target column from predictions file.")


def _top10_metrics(group: pd.DataFrame, target_col: str) -> tuple[float, float]:
    if group.empty:
        return float("nan"), float("nan")
    true_top10 = set(group.nsmallest(10, target_col).index)
    pred_top10 = set(group.nsmallest(10, "predicted_position").index)
    tp = len(true_top10 & pred_top10)
    denom = 10 if group.shape[0] >= 10 else max(group.shape[0], 1)
    precision = tp / denom
    recall = tp / denom
    return precision, recall


def _per_race_metrics(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["season", "round"], dropna=True)
    for (season, rnd), group in grouped:
        group = group.dropna(subset=[target_col, "predicted_position"])
        if group.empty:
            continue
        residuals = group["predicted_position"] - group[target_col]
        mae = float(np.mean(np.abs(residuals)))
        spearman = group[target_col].corr(
            group["predicted_position"], method="spearman"
        )
        precision, recall = _top10_metrics(group, target_col)
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
                "mae": mae,
                "spearman": spearman,
                "top10_precision": precision,
                "top10_recall": recall,
                "n_drivers": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _model_name_from_path(path: Path) -> str:
    return path.parent.name


def _summary_metrics(df: pd.DataFrame, target_col: str) -> dict:
    df = df.dropna(subset=[target_col, "predicted_position"])
    residuals = df["predicted_position"] - df[target_col]
    mae = float(np.mean(np.abs(residuals))) if len(residuals) else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else float("nan")
    spearman = df[target_col].corr(df["predicted_position"], method="spearman")
    per_race = df.groupby(["season", "round"], dropna=True)
    precisions = []
    recalls = []
    for _, group in per_race:
        precision, recall = _top10_metrics(group, target_col)
        if not np.isnan(precision):
            precisions.append(precision)
        if not np.isnan(recall):
            recalls.append(recall)
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "top10_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "top10_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "n_rows": int(df.shape[0]),
        "n_races": int(df[["season", "round"]].drop_duplicates().shape[0]),
    }


def _write_report(
    output_path: Path,
    summary: pd.DataFrame,
    per_race: pd.DataFrame,
    phase6_eval: pd.DataFrame,
) -> None:
    with output_path.open("w") as f:
        f.write("Phase 7 - Model Evaluation Report\n")
        f.write("===============================\n\n")
        if summary.empty:
            f.write("No predictions found. Run Phase 5 first.\n")
            return

        f.write("Summary Metrics (2025)\n")
        f.write("----------------------\n")
        for _, row in summary.iterrows():
            f.write(
                f"- {row['model']} | {row['target']}: "
                f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, "
                f"Spearman={row['spearman']:.3f}, "
                f"Top10 P={row['top10_precision']:.3f}, "
                f"Top10 R={row['top10_recall']:.3f}\n"
            )

        f.write("\nBest Models (by MAE)\n")
        f.write("---------------------\n")
        for target in summary["target"].unique():
            subset = summary[summary["target"] == target].copy()
            subset = subset.dropna(subset=["mae"])
            if subset.empty:
                continue
            best = subset.sort_values("mae").iloc[0]
            f.write(f"- {target}: {best['model']} (MAE={best['mae']:.3f})\n")

        f.write("\nWorst Races by MAE (Top 3 per Model/Target)\n")
        f.write("-------------------------------------------\n")
        if per_race.empty:
            f.write("No per-race metrics available.\n")
        else:
            grouped = per_race.sort_values("mae", ascending=False).groupby(
                ["model", "target"], dropna=True
            )
            for (model, target), group in grouped:
                f.write(f"- {model} | {target}:\n")
                worst = group.head(3)
                for _, row in worst.iterrows():
                    gp = row.get("grand_prix_name") or "Unknown GP"
                    f.write(
                        f"  {int(row['season'])} R{int(row['round']):02d} "
                        f"{gp} (MAE={row['mae']:.3f})\n"
                    )

        if not phase6_eval.empty:
            f.write("\nPhase 6 Evaluation (2026 Predictions)\n")
            f.write("-------------------------------------\n")
            for _, row in phase6_eval.iterrows():
                gp = row.get("grand_prix_name") or "Unknown GP"
                f.write(
                    f"- 2026 R{int(row['round']):02d} {gp}: "
                    f"MAE={row['mae']:.3f}, Spearman={row['spearman']:.3f}, "
                    f"Top10 P={row['top10_precision']:.3f}, "
                    f"Top10 R={row['top10_recall']:.3f}\n"
                )


def _plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summary.copy()
    summary["label"] = summary["model"] + " | " + summary["target"]
    summary = summary.sort_values("mae", ascending=True)

    plt.figure(figsize=(9, 4.5))
    plt.bar(summary["label"], summary["mae"], color="#2E86AB")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE (positions)")
    plt.title("Model MAE by Target (2025)")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "mae_by_model_target_2025.png")
    plt.close()


def _load_2026_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in results_dir.glob("2026/**/R/results.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        driver_key = None
        for col in ["DriverNumber", "Abbreviation", "Driver", "FullName"]:
            if col in df.columns:
                driver_key = col
                break
        if driver_key is None:
            continue
        if "FullName" in df.columns:
            driver_name = df["FullName"]
        elif "Abbreviation" in df.columns:
            driver_name = df["Abbreviation"]
        elif "Driver" in df.columns:
            driver_name = df["Driver"]
        else:
            driver_name = df[driver_key]

        out = pd.DataFrame(
            {
                "season": pd.to_numeric(df.get("Season", pd.NA), errors="coerce"),
                "round": pd.to_numeric(df.get("RoundNumber", pd.NA), errors="coerce"),
                "grand_prix_name": df.get("EventName", pd.NA),
                "driver_name": driver_name,
                "race_position": pd.to_numeric(
                    df.get("Position", pd.NA), errors="coerce"
                ),
            }
        )
        frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "grand_prix_name",
                "driver_name",
                "race_position",
            ]
        )
    results = pd.concat(frames, ignore_index=True)
    results = results.dropna(subset=["season", "round", "race_position"])
    results["season"] = results["season"].astype("Int64")
    results["round"] = results["round"].astype("Int64")
    return results


def _evaluate_phase6_predictions(models_dir: Path, output_dir: Path) -> pd.DataFrame:
    prediction_files = sorted((models_dir / "ensemble").glob("predict_2026_round*.csv"))
    if not prediction_files:
        return pd.DataFrame()

    results = _load_2026_results(PROJECT_ROOT / "data" / "raw" / "fastf1")
    if results.empty:
        return pd.DataFrame()

    rows = []
    for path in prediction_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        round_token = path.stem.split("_")[-1]
        round_token = round_token.replace("round", "")
        try:
            rnd = int(round_token)
        except ValueError:
            continue

        actual = results[results["round"] == rnd].copy()
        if actual.empty:
            continue

        driver_col = "Driver" if "Driver" in df.columns else "driver_name"
        pred_col = "Predicted Finish Position"
        if pred_col not in df.columns or driver_col not in df.columns:
            continue

        merged = actual.merge(
            df[[driver_col, pred_col]],
            left_on="driver_name",
            right_on=driver_col,
            how="inner",
        )
        if merged.empty:
            continue

        merged["predicted_position"] = pd.to_numeric(merged[pred_col], errors="coerce")
        merged = merged.dropna(subset=["predicted_position", "race_position"])
        if merged.empty:
            continue

        residuals = merged["predicted_position"] - merged["race_position"]
        mae = float(np.mean(np.abs(residuals)))
        spearman = merged["race_position"].corr(
            merged["predicted_position"], method="spearman"
        )
        precision, recall = _top10_metrics(merged, "race_position")
        gp = merged["grand_prix_name"].iloc[0] if "grand_prix_name" in merged else None

        rows.append(
            {
                "season": 2026,
                "round": int(rnd),
                "grand_prix_name": gp,
                "mae": mae,
                "spearman": spearman,
                "top10_precision": precision,
                "top10_recall": recall,
                "n_drivers": int(merged.shape[0]),
            }
        )

    phase6_eval = pd.DataFrame(rows)
    if phase6_eval.empty:
        return phase6_eval

    phase6_eval = phase6_eval.sort_values(["season", "round"])
    phase6_eval.to_csv(output_dir / "phase6_eval_2026.csv", index=False)
    return phase6_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: Evaluate model performance")
    parser.add_argument(
        "--predictions-dir",
        default=str(PROJECT_ROOT / "models"),
        help="Directory containing model prediction CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "evaluation"),
        help="Directory to write evaluation outputs",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=50,
        help="Number of largest errors to save per model/target",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = _find_prediction_files(predictions_dir)
    if not prediction_files:
        print("No prediction files found. Run Phase 5 training first.")
        return

    summary_rows = []
    per_race_rows = []
    driver_rows = []
    error_rows = []

    for path in prediction_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        model_name = _model_name_from_path(path)
        target_label, target_col = _infer_target(df)

        metrics = _summary_metrics(df, target_col)
        summary_rows.append(
            {
                "model": model_name,
                "target": target_label,
                **metrics,
            }
        )

        per_race = _per_race_metrics(df, target_col)
        if not per_race.empty:
            per_race.insert(0, "target", target_label)
            per_race.insert(0, "model", model_name)
            per_race_rows.append(per_race)

        if "driver_name" in df.columns:
            residuals = df["predicted_position"] - df[target_col]
            driver_summary = (
                df.assign(
                    abs_error=np.abs(residuals),
                    error=residuals,
                )
                .groupby("driver_name", dropna=True)
                .agg(
                    avg_abs_error=("abs_error", "mean"),
                    mean_error=("error", "mean"),
                    n_rows=("error", "size"),
                )
                .reset_index()
            )
            driver_summary.insert(0, "target", target_label)
            driver_summary.insert(0, "model", model_name)
            driver_rows.append(driver_summary)

        if "driver_name" in df.columns:
            df_errors = df.copy()
            df_errors["error"] = df_errors["predicted_position"] - df_errors[target_col]
            df_errors["abs_error"] = df_errors["error"].abs()
            top_errors = df_errors.nlargest(args.top_errors, "abs_error")
            if not top_errors.empty:
                top_errors = top_errors[
                    [
                        "driver_name",
                        "team",
                        "season",
                        "round",
                        target_col,
                        "predicted_position",
                        "error",
                        "abs_error",
                    ]
                ].copy()
                top_errors.insert(0, "target", target_label)
                top_errors.insert(0, "model", model_name)
                error_rows.append(top_errors)

    summary_df = pd.DataFrame(summary_rows)
    per_race_df = (
        pd.concat(per_race_rows, ignore_index=True) if per_race_rows else pd.DataFrame()
    )
    driver_df = (
        pd.concat(driver_rows, ignore_index=True) if driver_rows else pd.DataFrame()
    )
    error_df = (
        pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame()
    )

    summary_path = output_dir / "summary_metrics_2025.csv"
    per_race_path = output_dir / "per_race_metrics_2025.csv"
    driver_path = output_dir / "driver_error_summary_2025.csv"
    error_path = output_dir / "largest_errors_2025.csv"
    report_path = output_dir / "report_phase7.txt"

    summary_df.to_csv(summary_path, index=False)
    if not per_race_df.empty:
        per_race_df.to_csv(per_race_path, index=False)
    if not driver_df.empty:
        driver_df.to_csv(driver_path, index=False)
    if not error_df.empty:
        error_df.to_csv(error_path, index=False)

    phase6_eval = _evaluate_phase6_predictions(predictions_dir, output_dir)
    _write_report(report_path, summary_df, per_race_df, phase6_eval)
    _plot_summary(summary_df, output_dir / "plots")

    print(f"Saved summary -> {summary_path}")
    if not per_race_df.empty:
        print(f"Saved per-race metrics -> {per_race_path}")
    if not driver_df.empty:
        print(f"Saved driver error summary -> {driver_path}")
    if not error_df.empty:
        print(f"Saved largest errors -> {error_path}")
    print(f"Saved report -> {report_path}")
    if not phase6_eval.empty:
        print(f"Saved Phase 6 eval -> {output_dir / 'phase6_eval_2026.csv'}")


if __name__ == "__main__":
    main()
