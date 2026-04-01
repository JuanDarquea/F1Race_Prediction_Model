import argparse
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
TRACK_TYPE_PATH = PROJECT_ROOT / "data" / "track_types.csv"


def _find_result_files(session_code: str) -> List[Path]:
    return sorted(RAW_DIR.glob(f"*/**/{session_code}/results.csv"))


def _find_lap_files(session_code: str) -> List[Path]:
    return sorted(RAW_DIR.glob(f"*/**/{session_code}/laps.csv"))


def _pick_driver_key(df: pd.DataFrame) -> str:
    for col in ["DriverNumber", "Abbreviation", "Driver", "FullName"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _driver_name_from_df(df: pd.DataFrame, driver_key: str) -> pd.Series:
    if "FullName" in df.columns:
        return df["FullName"]
    if "Abbreviation" in df.columns:
        return df["Abbreviation"]
    if "Driver" in df.columns:
        return df["Driver"]
    return df[driver_key]


def _load_results(session_code: str) -> pd.DataFrame:
    frames = []
    for path in _find_result_files(session_code):
        df = pd.read_csv(path)
        if df.empty:
            continue
        driver_key = _pick_driver_key(df)
        df = df.copy()
        df["driver_id"] = df[driver_key]
        df["driver_name"] = _driver_name_from_df(df, driver_key)
        df["team"] = df.get("TeamName", df.get("Team", pd.NA))
        df["track"] = df.get("EventName", pd.NA)
        df["season"] = df.get("Season", pd.NA)
        df["round"] = df.get("RoundNumber", pd.NA)
        df["position"] = df.get("Position", pd.NA)
        df["points"] = df.get("Points", 0)
        df["status"] = df.get("Status", pd.NA)
        frames.append(
            df[
                [
                    "season",
                    "round",
                    "track",
                    "driver_name",
                    "driver_id",
                    "team",
                    "position",
                    "points",
                    "status",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "track",
                "driver_name",
                "driver_id",
                "team",
                "position",
                "points",
                "status",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0)
    return out


def _lap_time_seconds(df: pd.DataFrame) -> pd.Series:
    if "LapTimeSeconds" in df.columns:
        return pd.to_numeric(df["LapTimeSeconds"], errors="coerce")
    if "LapTime" in df.columns:
        return pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds()
    return pd.Series([pd.NA] * len(df))


def _load_practice_pace() -> pd.DataFrame:
    frames = []
    for session_code in ["FP1", "FP2", "FP3"]:
        for path in _find_lap_files(session_code):
            df = pd.read_csv(path)
            if df.empty:
                continue
            driver_key = _pick_driver_key(df)
            df = df.copy()
            df["driver_name"] = _driver_name_from_df(df, driver_key)
            df["season"] = df.get("Season", pd.NA)
            df["round"] = df.get("RoundNumber", pd.NA)
            df["track"] = df.get("EventName", pd.NA)
            df["lap_time"] = _lap_time_seconds(df)
            best = (
                df.dropna(subset=["lap_time"])
                .groupby(["season", "round", "track", "driver_name"], dropna=True)
                .agg(practice_pace=("lap_time", "min"))
                .reset_index()
            )
            frames.append(best)

    if not frames:
        return pd.DataFrame(
            columns=["season", "round", "track", "driver_name", "practice_pace"]
        )

    all_practice = pd.concat(frames, ignore_index=True)
    all_practice["season"] = pd.to_numeric(
        all_practice["season"], errors="coerce"
    ).astype("Int64")
    all_practice["round"] = pd.to_numeric(
        all_practice["round"], errors="coerce"
    ).astype("Int64")

    best_overall = (
        all_practice.groupby(["season", "round", "track", "driver_name"], dropna=True)
        .agg(practice_pace=("practice_pace", "min"))
        .reset_index()
    )
    return best_overall


def _load_track_types(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["track", "track_type"]).to_csv(path, index=False)
        print(f"Created track type template at {path}")
        return pd.DataFrame(columns=["track", "track_type"])

    df = pd.read_csv(path)
    if "track" not in df.columns or "track_type" not in df.columns:
        raise ValueError(
            "track_types.csv must include 'track' and 'track_type' columns"
        )
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round"]).copy()

    df["avg_finish_last_5"] = df.groupby("driver_name", dropna=True)[
        "race_position"
    ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    df["avg_finish_last_10"] = df.groupby("driver_name", dropna=True)[
        "race_position"
    ].transform(lambda s: s.rolling(10, min_periods=1).mean().shift(1))
    df["avg_qualifying_position"] = df.groupby("driver_name", dropna=True)[
        "qualifying_position"
    ].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
    df["points_last_races"] = df.groupby("driver_name", dropna=True)[
        "race_points"
    ].transform(lambda s: s.rolling(5, min_periods=1).sum().shift(1))

    df["team_avg_finish"] = df.groupby("team", dropna=True)["race_position"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1)
    )
    df["constructor_points"] = df.groupby("team", dropna=True)["race_points"].transform(
        lambda s: s.rolling(5, min_periods=1).sum().shift(1)
    )

    df["driver_performance_at_track"] = df.groupby(
        ["driver_name", "track"], dropna=True
    )["race_position"].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))

    return df


def build_feature_dataset(track_type_path: Path) -> pd.DataFrame:
    race_results = _load_results("R")
    qual_results = _load_results("Q")
    sprint_results = _load_results("S")
    sprint_qual_results = _load_results("SQ")
    practice = _load_practice_pace()
    track_types = _load_track_types(track_type_path)

    race = race_results.rename(
        columns={
            "position": "race_position",
            "points": "race_points",
        }
    )
    qual = qual_results.rename(columns={"position": "qualifying_position"})
    sprint = sprint_results.rename(
        columns={
            "position": "sprint_position",
            "points": "sprint_points",
        }
    )
    sprint_qual = sprint_qual_results.rename(
        columns={"position": "sprint_qualifying_position"}
    )

    base = race.merge(
        qual[["season", "round", "driver_name", "qualifying_position"]],
        on=["season", "round", "driver_name"],
        how="left",
    )
    base = base.merge(
        sprint[["season", "round", "driver_name", "sprint_position", "sprint_points"]],
        on=["season", "round", "driver_name"],
        how="left",
    )
    base = base.merge(
        sprint_qual[["season", "round", "driver_name", "sprint_qualifying_position"]],
        on=["season", "round", "driver_name"],
        how="left",
    )
    base = base.merge(
        practice[["season", "round", "driver_name", "practice_pace"]],
        on=["season", "round", "driver_name"],
        how="left",
    )

    if not track_types.empty:
        base = base.merge(track_types, on="track", how="left")
    else:
        base["track_type"] = "unknown"

    base = _add_rolling_features(base)

    base["season"] = base["season"].astype("Int64")
    base["round"] = base["round"].astype("Int64")
    base["race_position"] = pd.to_numeric(base["race_position"], errors="coerce")
    base["qualifying_position"] = pd.to_numeric(
        base["qualifying_position"], errors="coerce"
    )
    base["sprint_position"] = pd.to_numeric(base["sprint_position"], errors="coerce")
    base["sprint_points"] = pd.to_numeric(base["sprint_points"], errors="coerce")
    base["sprint_qualifying_position"] = pd.to_numeric(
        base["sprint_qualifying_position"], errors="coerce"
    )
    base["practice_pace"] = pd.to_numeric(base["practice_pace"], errors="coerce")

    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Feature Engineering")
    parser.add_argument(
        "--output",
        default=str(FEATURE_DIR / "feature_dataset.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--track-type-path",
        default=str(TRACK_TYPE_PATH),
        help="CSV file with columns track,track_type",
    )
    args = parser.parse_args()

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    feature_df = build_feature_dataset(Path(args.track_type_path))
    feature_df.to_csv(args.output, index=False)
    print(f"Saved feature dataset -> {args.output} ({len(feature_df)} rows)")


if __name__ == "__main__":
    main()
