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


def _find_weather_files(session_code: str) -> List[Path]:
    return sorted(RAW_DIR.glob(f"*/**/{session_code}/weather.csv"))


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


def _build_name_map(lap_path: Path) -> dict:
    """Build a mapping from driver abbreviation/number to FullName using results.csv."""
    results_path = lap_path.parent / "results.csv"
    if not results_path.exists():
        return {}
    try:
        results = pd.read_csv(results_path)
    except Exception:
        return {}
    name_map: dict = {}
    if "FullName" in results.columns:
        if "Abbreviation" in results.columns:
            for abbr, full in zip(
                results["Abbreviation"].astype(str), results["FullName"].astype(str)
            ):
                name_map[abbr] = full
        if "DriverNumber" in results.columns:
            for num, full in zip(
                results["DriverNumber"].astype(str), results["FullName"].astype(str)
            ):
                name_map[num] = full
        if "Driver" in results.columns:
            for drv, full in zip(
                results["Driver"].astype(str), results["FullName"].astype(str)
            ):
                name_map[drv] = full
    return name_map


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


def _load_weather_features() -> pd.DataFrame:
    _WEATHER_COLS = [
        "season",
        "round",
        "track",
        "race_air_temp",
        "race_track_temp",
        "race_humidity",
        "race_rainfall",
        "race_is_wet",
        "temp_delta_quali_race",
    ]

    frames = []
    for session_code in ["R", "Q"]:
        for path in _find_weather_files(session_code):
            df = pd.read_csv(path)
            if df.empty:
                continue
            first = df.iloc[0]
            season = first.get("Season", pd.NA)
            round_num = first.get("RoundNumber", pd.NA)
            track = first.get("EventName", pd.NA)
            session_name = first.get("SessionName", session_code)

            row = {
                "season": season,
                "round": round_num,
                "track": track,
                "session": session_name,
                "air_temp": pd.to_numeric(df.get("AirTemp"), errors="coerce").median(),
                "track_temp": pd.to_numeric(
                    df.get("TrackTemp"), errors="coerce"
                ).median(),
                "humidity": pd.to_numeric(df.get("Humidity"), errors="coerce").median(),
                "rainfall": pd.to_numeric(df.get("Rainfall"), errors="coerce").median(),
            }
            frames.append(row)

    if not frames:
        return pd.DataFrame(columns=_WEATHER_COLS)

    raw = pd.DataFrame(frames)
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce").astype("Int64")
    raw["round"] = pd.to_numeric(raw["round"], errors="coerce").astype("Int64")

    race_wx = raw[raw["session"].str.contains("Race", case=False, na=False)].rename(
        columns={
            "air_temp": "race_air_temp",
            "track_temp": "race_track_temp",
            "humidity": "race_humidity",
            "rainfall": "race_rainfall",
        }
    )[
        [
            "season",
            "round",
            "track",
            "race_air_temp",
            "race_track_temp",
            "race_humidity",
            "race_rainfall",
        ]
    ]

    quali_wx = raw[raw["session"].str.contains("Qual", case=False, na=False)].rename(
        columns={"track_temp": "quali_track_temp"}
    )[["season", "round", "quali_track_temp"]]

    out = race_wx.merge(quali_wx, on=["season", "round"], how="left")

    out["race_air_temp"] = pd.to_numeric(out["race_air_temp"], errors="coerce")
    out["race_track_temp"] = pd.to_numeric(out["race_track_temp"], errors="coerce")
    out["race_humidity"] = pd.to_numeric(out["race_humidity"], errors="coerce")
    out["race_rainfall"] = pd.to_numeric(out["race_rainfall"], errors="coerce")
    out["race_is_wet"] = (out["race_rainfall"] > 0).astype("Int64")
    out["temp_delta_quali_race"] = out["race_track_temp"] - pd.to_numeric(
        out["quali_track_temp"], errors="coerce"
    )

    return out[_WEATHER_COLS]


def _load_tire_features() -> pd.DataFrame:
    _TIRE_COLS = [
        "season",
        "round",
        "track",
        "driver_name",
        "num_stints",
        "avg_tyre_life",
        "max_tyre_life",
        "primary_compound",
    ]

    frames = []
    for path in _find_lap_files("R"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "Stint" not in df.columns:
            continue
        driver_key = _pick_driver_key(df)
        df = df.copy()
        # Map to FullName using results.csv so names match _load_results()
        name_map = _build_name_map(path)
        raw_names = _driver_name_from_df(df, driver_key)
        df["driver_name"] = raw_names.astype(str).map(name_map).fillna(raw_names)
        season = df["Season"].iloc[0] if "Season" in df.columns else pd.NA
        round_num = df["RoundNumber"].iloc[0] if "RoundNumber" in df.columns else pd.NA
        track = df["EventName"].iloc[0] if "EventName" in df.columns else pd.NA

        grouped = df.groupby("driver_name", dropna=True)

        num_stints = grouped["Stint"].nunique().rename("num_stints")

        if "TyreLife" in df.columns:
            tyre_life_num = pd.to_numeric(df["TyreLife"], errors="coerce")
            df["_tyre_life"] = tyre_life_num
            avg_tyre_life = grouped["_tyre_life"].mean().rename("avg_tyre_life")
            max_tyre_life = grouped["_tyre_life"].max().rename("max_tyre_life")
        else:
            avg_tyre_life = pd.Series(
                pd.NA, index=num_stints.index, name="avg_tyre_life", dtype="float64"
            )
            max_tyre_life = pd.Series(
                pd.NA, index=num_stints.index, name="max_tyre_life", dtype="float64"
            )

        if "Compound" in df.columns:
            primary_compound = (
                grouped["Compound"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
                .rename("primary_compound")
            )
        else:
            primary_compound = pd.Series(
                pd.NA, index=num_stints.index, name="primary_compound", dtype="object"
            )

        race_df = pd.concat(
            [num_stints, avg_tyre_life, max_tyre_life, primary_compound], axis=1
        ).reset_index()
        race_df["season"] = season
        race_df["round"] = round_num
        race_df["track"] = track
        frames.append(race_df[_TIRE_COLS])

    if not frames:
        return pd.DataFrame(columns=_TIRE_COLS)

    out = pd.concat(frames, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    return out


def _load_pit_features() -> pd.DataFrame:
    _PIT_COLS = [
        "season",
        "round",
        "track",
        "driver_name",
        "avg_pit_time",
        "total_pit_time_lost",
    ]

    frames = []
    for path in _find_lap_files("R"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        driver_key = _pick_driver_key(df)
        df = df.copy()
        name_map = _build_name_map(path)
        raw_names = _driver_name_from_df(df, driver_key)
        df["driver_name"] = raw_names.astype(str).map(name_map).fillna(raw_names)
        season = df["Season"].iloc[0] if "Season" in df.columns else pd.NA
        round_num = df["RoundNumber"].iloc[0] if "RoundNumber" in df.columns else pd.NA
        track = df["EventName"].iloc[0] if "EventName" in df.columns else pd.NA

        # Pit stop duration = PitOutTime(next lap) - PitInTime(current lap)
        # Both columns are session-elapsed times, NOT durations.
        has_in = "PitInTimeSeconds" in df.columns
        has_out = "PitOutTimeSeconds" in df.columns
        if not (has_in and has_out):
            continue

        pit_durations: list = []
        for drv, drv_laps in df.groupby("driver_name", dropna=True):
            drv_laps = drv_laps.sort_values("LapNumber")
            pit_in = pd.to_numeric(drv_laps["PitInTimeSeconds"], errors="coerce")
            pit_out = pd.to_numeric(drv_laps["PitOutTimeSeconds"], errors="coerce")
            # pit_in on lap N, pit_out on lap N+1 → duration = pit_out[N+1] - pit_in[N]
            for i in range(len(drv_laps) - 1):
                p_in = pit_in.iloc[i]
                p_out = pit_out.iloc[i + 1]
                if pd.notna(p_in) and pd.notna(p_out) and p_out > p_in:
                    dur = p_out - p_in
                    if 10 < dur < 120:  # sanity: 10s-120s is a realistic pit stop
                        pit_durations.append(
                            {"driver_name": str(drv), "pit_duration": dur}
                        )

        if not pit_durations:
            continue

        pit_df = pd.DataFrame(pit_durations)
        grouped = pit_df.groupby("driver_name", dropna=True)
        avg_pit_time = grouped["pit_duration"].mean().rename("avg_pit_time")
        total_pit_time_lost = (
            grouped["pit_duration"].sum().rename("total_pit_time_lost")
        )

        race_df = pd.concat([avg_pit_time, total_pit_time_lost], axis=1).reset_index()
        race_df["season"] = season
        race_df["round"] = round_num
        race_df["track"] = track
        frames.append(race_df[_PIT_COLS])

    if not frames:
        return pd.DataFrame(columns=_PIT_COLS)

    out = pd.concat(frames, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    return out


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

    df["avg_finish_last_3"] = df.groupby("driver_name", dropna=True)[
        "race_position"
    ].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
    df["avg_finish_last_5"] = df.groupby("driver_name", dropna=True)[
        "race_position"
    ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    df["std_finish_last_5"] = df.groupby("driver_name", dropna=True)[
        "race_position"
    ].transform(lambda s: s.rolling(5, min_periods=2).std().shift(1))
    df["avg_finish_last_10"] = df.groupby("driver_name", dropna=True)[
        "race_position"
    ].transform(lambda s: s.rolling(10, min_periods=1).mean().shift(1))
    df["avg_qualifying_position"] = df.groupby("driver_name", dropna=True)[
        "qualifying_position"
    ].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
    df["avg_qualifying_last_5"] = df.groupby("driver_name", dropna=True)[
        "qualifying_position"
    ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    df["points_last_races"] = df.groupby("driver_name", dropna=True)[
        "race_points"
    ].transform(lambda s: s.rolling(5, min_periods=1).sum().shift(1))
    df["avg_points_last_5"] = df.groupby("driver_name", dropna=True)[
        "race_points"
    ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))

    df["team_avg_finish"] = df.groupby("team", dropna=True)["race_position"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1)
    )
    df["team_avg_qualifying"] = df.groupby("team", dropna=True)[
        "qualifying_position"
    ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    df["constructor_points"] = df.groupby("team", dropna=True)["race_points"].transform(
        lambda s: s.rolling(5, min_periods=1).sum().shift(1)
    )
    df["team_points_last_5"] = df.groupby("team", dropna=True)["race_points"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1)
    )

    race_minus_qual = df["race_position"] - df["qualifying_position"]
    df["avg_race_minus_qual_last_10"] = race_minus_qual.groupby(
        df["driver_name"], dropna=True
    ).transform(lambda s: s.rolling(10, min_periods=2).mean().shift(1))

    if "dnf_flag" in df.columns:
        df["driver_dnf_rate_last_10"] = df.groupby("driver_name", dropna=True)[
            "dnf_flag"
        ].transform(lambda s: s.rolling(10, min_periods=2).mean().shift(1))
        df["team_dnf_rate_last_10"] = df.groupby("team", dropna=True)[
            "dnf_flag"
        ].transform(lambda s: s.rolling(10, min_periods=2).mean().shift(1))

    df["driver_performance_at_track"] = df.groupby(
        ["driver_name", "track"], dropna=True
    )["race_position"].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
    df["team_performance_at_track"] = df.groupby(["team", "track"], dropna=True)[
        "race_position"
    ].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))

    # --- Rolling tire/pit features (Phase 8) ---
    if "num_stints" in df.columns:
        df["driver_avg_stints_last_5"] = df.groupby("driver_name", dropna=True)[
            "num_stints"
        ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))

    if "avg_pit_time" in df.columns:
        df["team_avg_pit_time_last_5"] = df.groupby("team", dropna=True)[
            "avg_pit_time"
        ].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))

    return df


def _impute_tiered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round"]).copy()
    team_group_cols = ["avg_pit_time", "total_pit_time_lost"]
    race_group_cols = [
        "race_air_temp",
        "race_track_temp",
        "race_humidity",
        "race_rainfall",
        "num_stints",
        "avg_tyre_life",
        "max_tyre_life",
    ]
    impute_log = {}
    for col in team_group_cols + race_group_cols:
        if col not in df.columns:
            continue
        before_na = df[col].isna().sum()
        if before_na == 0:
            continue
        if col in team_group_cols:
            group_median = df.groupby(["season", "round", "team"], dropna=True)[
                col
            ].transform("median")
        else:
            group_median = df.groupby(["season", "round"], dropna=True)[col].transform(
                "median"
            )
        df[col] = df[col].fillna(group_median)
        still_na = df[col].isna().sum()
        if still_na > 0:
            if col in team_group_cols:
                rolling_med = df.groupby("team", dropna=True)[col].transform(
                    lambda s: s.rolling(10, min_periods=1).median()
                )
            else:
                rolling_med = df.groupby("driver_name", dropna=True)[col].transform(
                    lambda s: s.rolling(10, min_periods=1).median()
                )
            df[col] = df[col].fillna(rolling_med)
        still_na = df[col].isna().sum()
        if still_na > 0:
            season_median = df.groupby("season", dropna=True)[col].transform("median")
            df[col] = df[col].fillna(season_median)
        after_na = df[col].isna().sum()
        imputed = before_na - after_na
        if imputed > 0:
            impute_log[col] = imputed
    if impute_log:
        print("Tiered imputation counts:")
        for col, count in sorted(impute_log.items()):
            print(f"  {col}: {count} values imputed")
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

    # --- New feature sources (Phase 8) ---
    weather = _load_weather_features()
    tire = _load_tire_features()
    pit = _load_pit_features()

    if not weather.empty:
        base = base.merge(weather, on=["season", "round", "track"], how="left")
    else:
        for col in [
            "race_air_temp",
            "race_track_temp",
            "race_humidity",
            "race_rainfall",
            "race_is_wet",
            "temp_delta_quali_race",
        ]:
            base[col] = pd.NA

    if not tire.empty:
        tire_merge = tire.drop(columns=["primary_compound"], errors="ignore")
        base = base.merge(
            tire_merge[
                [
                    "season",
                    "round",
                    "driver_name",
                    "num_stints",
                    "avg_tyre_life",
                    "max_tyre_life",
                ]
            ],
            on=["season", "round", "driver_name"],
            how="left",
        )
    else:
        for col in ["num_stints", "avg_tyre_life", "max_tyre_life"]:
            base[col] = pd.NA

    if not pit.empty:
        base = base.merge(
            pit[
                [
                    "season",
                    "round",
                    "driver_name",
                    "avg_pit_time",
                    "total_pit_time_lost",
                ]
            ],
            on=["season", "round", "driver_name"],
            how="left",
        )
    else:
        for col in ["avg_pit_time", "total_pit_time_lost"]:
            base[col] = pd.NA

    # Tiered imputation for new features
    base = _impute_tiered(base)

    if not track_types.empty:
        base = base.merge(track_types, on="track", how="left")
    else:
        base["track_type"] = "unknown"

    if "status" in base.columns:
        base["dnf_flag"] = base["status"].ne("Finished").astype("Int64")
    else:
        base["dnf_flag"] = pd.NA

    base["practice_pace_rank"] = base.groupby(["season", "round"], dropna=True)[
        "practice_pace"
    ].rank(method="min")
    base["practice_pace_percentile"] = base.groupby(["season", "round"], dropna=True)[
        "practice_pace"
    ].rank(pct=True)
    base["practice_pace_gap_to_best"] = base["practice_pace"] - base.groupby(
        ["season", "round"], dropna=True
    )["practice_pace"].transform("min")

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

    for col in [
        "race_air_temp",
        "race_track_temp",
        "race_humidity",
        "race_rainfall",
        "race_is_wet",
        "temp_delta_quali_race",
        "num_stints",
        "avg_tyre_life",
        "max_tyre_life",
        "avg_pit_time",
        "total_pit_time_lost",
        "driver_avg_stints_last_5",
        "team_avg_pit_time_last_5",
    ]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

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
