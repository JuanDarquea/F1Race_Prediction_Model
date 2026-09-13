"""Phase 11: Compare Phase 10's classifiers against Phase 5's regression-derived
top-10 numbers, side by side, without modifying anything Phase 5-8 produced.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_METRICS_PATH = PROJECT_ROOT / "data" / "evaluation" / "summary_metrics_2025.csv"
MODEL_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"


def load_regression_top10_baseline(summary_path: Path, target: str) -> Optional[dict]:
    """Return the best (lowest MAE) regression model's top-10 precision/recall for a target."""
    summary = pd.read_csv(summary_path)
    subset = summary[summary["target"] == target]
    if subset.empty:
        return None
    best = subset.loc[subset["mae"].idxmin()]
    # Suffixed with the season because summary_metrics_2025.csv is a single
    # 2023-24 -> 2025 evaluation, not a per-season series.
    return {
        "baseline_model_2025": best["model"],
        "baseline_top10_precision_2025": float(best["top10_precision"]),
        "baseline_top10_recall_2025": float(best["top10_recall"]),
    }


def build_comparison_table(
    top10_race_fold_metrics: pd.DataFrame,
    top10_qualifying_fold_metrics: pd.DataFrame,
    summary_path: Path = SUMMARY_METRICS_PATH,
) -> pd.DataFrame:
    """One row per (classifier, test_season): new classifier metrics next to the
    best regression-derived baseline for the same target.

    Two precision/recall pairs are reported per classifier. `classifier_precision`
    / `classifier_recall` use a fixed 0.5 probability threshold, which selects a
    variable number of drivers per race and is therefore NOT comparable to the
    baseline's numbers. `classifier_top10_cut_precision` / `_recall` use Phase 5's
    own selection rule (exactly as many picks as there are true top-10 finishers),
    and are the like-for-like comparison against the baseline columns.
    """
    rows = []
    for classifier, fold_metrics, target in (
        ("top10_race", top10_race_fold_metrics, "race"),
        ("top10_qualifying", top10_qualifying_fold_metrics, "qualifying"),
    ):
        baseline = load_regression_top10_baseline(summary_path, target)
        for _, fold in fold_metrics.iterrows():
            row = {
                "classifier": classifier,
                "test_season": fold["test_season"],
                "classifier_precision": fold["precision"],
                "classifier_recall": fold["recall"],
                "classifier_top10_cut_precision": fold["top10_cut_precision"],
                "classifier_top10_cut_recall": fold["top10_cut_recall"],
                "classifier_log_loss": fold["log_loss"],
                "classifier_brier_score": fold["brier_score"],
            }
            if baseline:
                row.update(baseline)
            rows.append(row)

    return pd.DataFrame(rows)


def _write_report(
    comparison: pd.DataFrame, winner_podium_metrics: pd.DataFrame, report_path: Path
) -> None:
    with report_path.open("w") as f:
        f.write("Phase 11 - Classifier Comparison Report\n")
        f.write("========================================\n\n")
        f.write(
            "Top-10 classifiers vs. best regression-derived baseline (by test season):\n\n"
        )
        f.write(comparison.to_string(index=False))
        f.write("\n\nNotes on reading this table:\n")
        f.write(
            "  - The baseline_*_2025 columns are a single fixed 2023-24 -> 2025 evaluation\n"
            "    repeated on every row, not re-computed per season, so only the 2025 rows\n"
            "    are a genuinely matched (same test season) comparison.\n"
        )
        f.write(
            "  - classifier_precision/recall use a fixed 0.5 probability threshold, which\n"
            "    picks a variable number of drivers per race; they are NOT comparable to the\n"
            "    baseline columns. classifier_top10_cut_precision/recall use Phase 5's own\n"
            "    selection rule (exactly as many picks as true top-10 finishers per race) and\n"
            "    are the like-for-like comparison. That rule forces precision == recall.\n"
        )
        f.write("\nWinner/Podium classifier fold metrics:\n\n")
        f.write(winner_podium_metrics.to_string(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 11: Compare Phase 10 classifiers vs Phase 5 baseline"
    )
    parser.add_argument("--summary-metrics", default=str(SUMMARY_METRICS_PATH))
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    top10_race_metrics = pd.read_csv(MODEL_DIR / "top10_race" / "fold_metrics.csv")
    top10_qualifying_metrics = pd.read_csv(
        MODEL_DIR / "top10_qualifying" / "fold_metrics.csv"
    )
    winner_podium_metrics = pd.read_csv(
        MODEL_DIR / "winner_podium" / "fold_metrics.csv"
    )

    comparison = build_comparison_table(
        top10_race_metrics, top10_qualifying_metrics, Path(args.summary_metrics)
    )
    comparison.to_csv(EVAL_DIR / "classifier_comparison.csv", index=False)
    _write_report(comparison, winner_podium_metrics, EVAL_DIR / "report_phase11.txt")
    print(f"Saved comparison -> {EVAL_DIR / 'classifier_comparison.csv'}")
    print(f"Saved report -> {EVAL_DIR / 'report_phase11.txt'}")


if __name__ == "__main__":
    main()
