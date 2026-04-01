import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "clean" / "fastf1_race_laps_clean.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"
EDA_DIR = PROJECT_ROOT / "data" / "eda"


def _find_result_files(session_code: str) -> List[Path]:
    return sorted(RAW_DIR.glob(f"*/**/{session_code}/results.csv"))


def _pick_driver_key(df: pd.DataFrame) -> str:
    for col in ["DriverNumber", "Abbreviation", "Driver", "FullName"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _load_results(session_code: str) -> pd.DataFrame:
    frames = []
    for path in _find_result_files(session_code):
        df = pd.read_csv(path)
        if df.empty:
            continue
        driver_key = _pick_driver_key(df)
        if "FullName" in df.columns:
            driver_name = df["FullName"]
        elif "Abbreviation" in df.columns:
            driver_name = df["Abbreviation"]
        elif "Driver" in df.columns:
            driver_name = df["Driver"]
        else:
            driver_name = df[driver_key]
        df = df.copy()
        df["driver_id"] = df[driver_key]
        df["driver_name"] = driver_name
        df["team"] = df.get("TeamName", df.get("Team", pd.NA))
        df["track"] = df.get("EventName", pd.NA)
        df["season"] = df.get("Season", pd.NA)
        df["round"] = df.get("RoundNumber", pd.NA)
        df["position"] = df.get("Position", pd.NA)
        df["status"] = df.get("Status", pd.NA)
        frames.append(
            df[
                [
                    "driver_id",
                    "driver_name",
                    "team",
                    "track",
                    "season",
                    "round",
                    "position",
                    "status",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "driver_id",
                "driver_name",
                "team",
                "track",
                "season",
                "round",
                "position",
                "status",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    return out


def _infer_dnf(status: pd.Series, position: pd.Series) -> pd.Series:
    finished_mask = status.astype("string").str.contains(
        "Finished|Lap", case=False, na=False
    )
    position_mask = pd.to_numeric(position, errors="coerce").notna()
    return ~(finished_mask | position_mask)


def analyze_average_finish(race_results: pd.DataFrame) -> pd.DataFrame:
    race_results = race_results.copy()
    race_results["dnf"] = _infer_dnf(race_results["status"], race_results["position"])
    avg_finish = (
        race_results.dropna(subset=["position"])
        .groupby("driver_name", dropna=True)
        .agg(
            team=("team", "first"),
            driver_id=("driver_id", "first"),
            races=("round", "nunique"),
            avg_position=("position", "mean"),
            median_position=("position", "median"),
            dnfs=("dnf", "sum"),
        )
        .sort_values("avg_position")
        .reset_index()
    )
    return avg_finish


def analyze_team_trends(race_results: pd.DataFrame) -> pd.DataFrame:
    race_results = race_results.copy()
    race_results = race_results.dropna(subset=["position", "season", "team"])
    team_trends = (
        race_results.groupby(["season", "team"], dropna=True)
        .agg(avg_position=("position", "mean"), races=("round", "nunique"))
        .reset_index()
        .sort_values(["season", "avg_position"])
    )
    return team_trends


def analyze_track_difficulty(
    race_results: pd.DataFrame, laps: pd.DataFrame
) -> pd.DataFrame:
    race_results = race_results.copy()
    race_results["dnf"] = _infer_dnf(race_results["status"], race_results["position"])
    dnf_rates = (
        race_results.groupby("track", dropna=True)
        .agg(dnf_rate=("dnf", "mean"), races=("round", "nunique"))
        .reset_index()
    )

    lap_stats = (
        laps.dropna(subset=["lap_time", "track"])
        .groupby("track", dropna=True)
        .agg(avg_lap_time=("lap_time", "mean"), lap_time_std=("lap_time", "std"))
        .reset_index()
    )

    track_difficulty = dnf_rates.merge(lap_stats, on="track", how="left")
    return track_difficulty.sort_values(["dnf_rate", "avg_lap_time"], ascending=False)


def plot_finish_positions(avg_finish: pd.DataFrame, out_path: Path, title: str) -> None:
    top = avg_finish.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top["driver_name"], top["avg_position"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Average Position (Lower is Better)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_lap_times(laps: pd.DataFrame, out_path: Path) -> None:
    sample = (
        laps.dropna(subset=["lap_time", "driver"])
        .groupby("driver", dropna=True)
        .head(200)
    )
    plt.figure(figsize=(10, 6))
    plt.hist(sample["lap_time"], bins=50, color="steelblue", alpha=0.8)
    plt.title("Lap Time Distribution (Sampled)")
    plt.xlabel("Lap Time (seconds)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_qualifying_vs_race(
    q_results: pd.DataFrame, r_results: pd.DataFrame, out_path: Path
) -> float:
    q_results = q_results.dropna(subset=["position", "season", "round", "driver_id"])
    r_results = r_results.dropna(subset=["position", "season", "round", "driver_id"])

    merged = q_results.merge(
        r_results,
        on=["season", "round", "driver_id"],
        suffixes=("_qual", "_race"),
        how="inner",
    )

    corr = merged["position_qual"].corr(merged["position_race"], method="spearman")

    plt.figure(figsize=(8, 6))
    plt.scatter(merged["position_qual"], merged["position_race"], alpha=0.4)
    plt.title("Qualifying vs Race Finish Position")
    plt.xlabel("Qualifying Position")
    plt.ylabel("Race Finish Position")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return corr


def plot_sprint_vs_race(
    sprint_results: pd.DataFrame, race_results: pd.DataFrame, out_path: Path
) -> float:
    sprint_results = sprint_results.dropna(
        subset=["position", "season", "round", "driver_id"]
    )
    race_results = race_results.dropna(
        subset=["position", "season", "round", "driver_id"]
    )

    merged = sprint_results.merge(
        race_results,
        on=["season", "round", "driver_id"],
        suffixes=("_sprint", "_race"),
        how="inner",
    )

    if merged.empty:
        return float("nan")

    corr = merged["position_sprint"].corr(merged["position_race"], method="spearman")

    plt.figure(figsize=(8, 6))
    plt.scatter(merged["position_sprint"], merged["position_race"], alpha=0.4)
    plt.title("Sprint vs Race Finish Position")
    plt.xlabel("Sprint Finish Position")
    plt.ylabel("Race Finish Position")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Exploratory data analysis")
    parser.add_argument(
        "--cleaned-path",
        default=str(CLEAN_PATH),
        help="Path to cleaned race lap dataset",
    )
    args = parser.parse_args()

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(args.cleaned_path).exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run phase2_data_cleaning.py first."
        )

    laps = pd.read_csv(args.cleaned_path)
    race_results = _load_results("R")
    qual_results = _load_results("Q")
    sprint_results = _load_results("S")

    avg_finish = analyze_average_finish(race_results)
    team_trends = analyze_team_trends(race_results)
    track_difficulty = analyze_track_difficulty(race_results, laps)
    avg_sprint_finish = analyze_average_finish(sprint_results)

    avg_finish.to_csv(EDA_DIR / "avg_finish_by_driver.csv", index=False)
    team_trends.to_csv(EDA_DIR / "team_trends.csv", index=False)
    track_difficulty.to_csv(EDA_DIR / "track_difficulty.csv", index=False)
    avg_sprint_finish.to_csv(EDA_DIR / "avg_sprint_finish_by_driver.csv", index=False)

    plot_finish_positions(
        avg_finish,
        EDA_DIR / "avg_finish_positions.png",
        "Average Race Finish Position (Top 15)",
    )
    if not avg_sprint_finish.empty:
        plot_finish_positions(
            avg_sprint_finish,
            EDA_DIR / "avg_sprint_finish_positions.png",
            "Average Sprint Finish Position (Top 15)",
        )
    plot_lap_times(laps, EDA_DIR / "lap_time_distribution.png")
    spearman_corr = plot_qualifying_vs_race(
        qual_results,
        race_results,
        EDA_DIR / "qual_vs_race_scatter.png",
    )
    sprint_corr = plot_sprint_vs_race(
        sprint_results, race_results, EDA_DIR / "sprint_vs_race_scatter.png"
    )

    consistency = (
        race_results.dropna(subset=["position"])
        .groupby("driver_name", dropna=True)
        .agg(
            driver_id=("driver_id", "first"),
            position_std=("position", "std"),
            races=("round", "nunique"),
        )
        .sort_values("position_std")
        .reset_index()
    )
    consistency.to_csv(EDA_DIR / "driver_consistency.csv", index=False)

    summary_path = EDA_DIR / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Phase 3 EDA Summary\n")
        f.write("===================\n\n")
        f.write(f"Qualifying vs Race Spearman correlation: {spearman_corr:.3f}\n")
        if pd.isna(sprint_corr):
            f.write("Sprint vs Race Spearman correlation: n/a (no sprint data)\n")
        else:
            f.write(f"Sprint vs Race Spearman correlation: {sprint_corr:.3f}\n")
        f.write("Lower position_std means more consistent finishes.\n")

    print(f"EDA outputs saved in {EDA_DIR}")


if __name__ == "__main__":
    main()
