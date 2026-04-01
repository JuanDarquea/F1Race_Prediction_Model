import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def _find_lap_files(session_codes: List[str]) -> List[Path]:
    lap_files: List[Path] = []
    for code in session_codes:
        lap_files.extend(RAW_DIR.glob(f"*/**/{code}/laps.csv"))
    return sorted(lap_files)


def _pick_driver_key(df: pd.DataFrame) -> str:
    if "DriverNumber" in df.columns:
        return "DriverNumber"
    if "Driver" in df.columns:
        return "Driver"
    if "Abbreviation" in df.columns:
        return "Abbreviation"
    return df.columns[0]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    driver_key = _pick_driver_key(df)
    lap_time_col = "LapTimeSeconds" if "LapTimeSeconds" in df.columns else "LapTime"

    if "FullName" in df.columns:
        driver_name = df["FullName"]
    elif "Abbreviation" in df.columns:
        driver_name = df["Abbreviation"]
    elif "Driver" in df.columns:
        driver_name = df["Driver"]
    else:
        driver_name = df[driver_key]

    standardized = pd.DataFrame(
        {
            "driver": df[driver_key],
            "driver_name": driver_name,
            "team": df.get("Team", pd.NA),
            "track": df.get("EventName", pd.NA),
            "grand_prix": df.get("EventName", pd.NA),
            "season": df.get("Season", pd.NA),
            "round": df.get("RoundNumber", pd.NA),
            "session": df.get("SessionName", "Race"),
            "position": df.get("Position", pd.NA),
            "lap_time": df.get(lap_time_col, pd.NA),
            "lap_number": df.get("LapNumber", pd.NA),
        }
    )

    standardized["driver_key"] = df[driver_key]
    return standardized


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["driver"] = df["driver"].astype("string")
    df["driver_name"] = df["driver_name"].astype("string")
    df["team"] = df["team"].astype("string")
    df["track"] = df["track"].astype("string")
    df["grand_prix"] = df["grand_prix"].astype("string")
    df["session"] = df["session"].astype("string")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df["lap_number"] = pd.to_numeric(df["lap_number"], errors="coerce").astype("Int64")
    df["dnf"] = df["dnf"].astype("boolean")
    return df


def _clean_missing(df: pd.DataFrame, drop_missing: bool) -> pd.DataFrame:
    if drop_missing:
        # Remove rows with missing lap times or positions (common for DNFs or incomplete laps)
        return df.dropna(subset=["lap_time", "position"])

    # If we keep missing rows, at least normalize the NaNs
    return df.copy()


def _load_results_for_laps(lap_path: Path) -> Optional[pd.DataFrame]:
    results_path = lap_path.parent / "results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)


def _infer_dnf(results: pd.DataFrame, driver_key: str) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame(columns=[driver_key, "dnf"])

    status = results.get("Status")
    position = results.get("Position")

    if status is not None:
        finished_mask = status.astype("string").str.contains(
            "Finished|Lap", case=False, na=False
        )
    else:
        finished_mask = pd.Series([False] * len(results))

    if position is not None:
        position_mask = pd.to_numeric(position, errors="coerce").notna()
    else:
        position_mask = pd.Series([False] * len(results))

    dnf = ~(finished_mask | position_mask)
    return pd.DataFrame({"driver_key": results[driver_key], "dnf": dnf})


def _infer_driver_name(results: pd.DataFrame, driver_key: str) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame(columns=["driver_key", "driver_name"])

    if "FullName" in results.columns:
        driver_name = results["FullName"]
    elif "Abbreviation" in results.columns:
        driver_name = results["Abbreviation"]
    elif "Driver" in results.columns:
        driver_name = results["Driver"]
    else:
        driver_name = results[driver_key]

    return pd.DataFrame({"driver_key": results[driver_key], "driver_name": driver_name})


def _build_from_lap_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    standardized = _standardize_columns(raw)
    driver_key = _pick_driver_key(raw)

    results = _load_results_for_laps(path)
    if results is not None and driver_key in results.columns:
        dnf_map = _infer_dnf(results, driver_key)
        standardized = standardized.merge(dnf_map, on="driver_key", how="left")
        name_map = _infer_driver_name(results, driver_key)
        standardized = standardized.merge(
            name_map, on="driver_key", how="left", suffixes=("", "_results")
        )
        if "driver_name_results" in standardized.columns:
            standardized["driver_name"] = standardized[
                "driver_name_results"
            ].combine_first(standardized["driver_name"])
            standardized.drop(columns=["driver_name_results"], inplace=True)
    else:
        standardized["dnf"] = pd.NA

    standardized.drop(columns=["driver_key"], inplace=True)
    return standardized


def build_clean_dataset(
    drop_missing: bool = True, session_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    if session_codes is None:
        session_codes = ["R", "S"]
    lap_files = _find_lap_files(session_codes)
    if not lap_files:
        raise FileNotFoundError(
            "No lap files found for the requested sessions. "
            "Run phase1_data_collection.py first to populate data/raw/fastf1."
        )

    frames = []
    for path in lap_files:
        frames.append(_build_from_lap_file(path))

    combined = pd.concat(frames, ignore_index=True)
    combined = _coerce_types(combined)
    combined = _clean_missing(combined, drop_missing=drop_missing)

    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: clean and combine race/sprint lap data"
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep rows with missing lap_time or position (useful for inspection).",
    )
    parser.add_argument(
        "--split-by-year",
        action="store_true",
        help="Also write one cleaned CSV per season.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["R", "S"],
        help="Session codes to include (default: R S).",
    )
    parser.add_argument(
        "--aggregate-by-driver",
        action="store_true",
        help="Write a per-driver aggregated dataset.",
    )
    parser.add_argument(
        "--aggregate-by-circuit",
        action="store_true",
        help="Write a per-circuit aggregated dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(CLEAN_DIR / "fastf1_race_laps_clean.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = build_clean_dataset(
        drop_missing=not args.keep_missing, session_codes=args.sessions
    )
    cleaned.to_csv(args.output, index=False)
    print(f"Saved cleaned dataset -> {args.output} ({len(cleaned)} rows)")

    if args.split_by_year:
        output_path = Path(args.output)
        for year, subset in cleaned.groupby("season", dropna=True):
            if pd.isna(year):
                continue
            year_int = int(year)
            year_path = output_path.with_name(
                f"{output_path.stem}_{year_int}{output_path.suffix}"
            )
            subset.to_csv(year_path, index=False)
            print(f"Saved cleaned dataset -> {year_path} ({len(subset)} rows)")

    output_path = Path(args.output)
    cleaned_non_null = cleaned.dropna(subset=["lap_time"])

    if args.aggregate_by_driver:
        by_driver = (
            cleaned_non_null.groupby("driver_name", dropna=True)
            .agg(
                driver_id=("driver", "first"),
                team=("team", "first"),
                seasons=("season", "nunique"),
                races=("round", "nunique"),
                laps=("lap_time", "count"),
                avg_lap_time=("lap_time", "mean"),
                best_lap_time=("lap_time", "min"),
                median_lap_time=("lap_time", "median"),
                lap_time_std=("lap_time", "std"),
                dnfs=("dnf", "sum"),
            )
            .reset_index()
        )
        by_driver_path = output_path.with_name(
            f"{output_path.stem}_by_driver{output_path.suffix}"
        )
        by_driver.to_csv(by_driver_path, index=False)
        print(f"Saved aggregated dataset -> {by_driver_path} ({len(by_driver)} rows)")

    if args.aggregate_by_circuit:
        by_circuit = (
            cleaned_non_null.groupby("track", dropna=True)
            .agg(
                seasons=("season", "nunique"),
                races=("round", "nunique"),
                drivers=("driver", "nunique"),
                laps=("lap_time", "count"),
                avg_lap_time=("lap_time", "mean"),
                best_lap_time=("lap_time", "min"),
                median_lap_time=("lap_time", "median"),
                lap_time_std=("lap_time", "std"),
                dnfs=("dnf", "sum"),
            )
            .reset_index()
        )
        by_circuit_path = output_path.with_name(
            f"{output_path.stem}_by_circuit{output_path.suffix}"
        )
        by_circuit.to_csv(by_circuit_path, index=False)
        print(f"Saved aggregated dataset -> {by_circuit_path} ({len(by_circuit)} rows)")


if __name__ == "__main__":
    main()
