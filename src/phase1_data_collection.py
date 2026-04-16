import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import fastf1
import pandas as pd
from fastf1 import exceptions as f1_exceptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "fastf1_cache"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds()


def _normalize_laps(laps: pd.DataFrame, session_meta: dict) -> pd.DataFrame:
    df = laps.copy()

    # Convert timedelta columns to seconds for easier ML usage.
    for col in [
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "PitInTime",
        "PitOutTime",
    ]:
        if col in df.columns and pd.api.types.is_timedelta64_dtype(df[col]):
            df[f"{col}Seconds"] = _to_seconds(df[col])

    for key, value in session_meta.items():
        df[key] = value

    return df


def _session_meta(session) -> dict:
    event = session.event
    try:
        session_type = str(session.session_info)
    except f1_exceptions.DataNotLoadedError:
        session_type = str(session.name)
    return {
        "Season": int(event["EventDate"].year),
        "RoundNumber": int(event["RoundNumber"]),
        "EventName": str(event["EventName"]),
        "SessionName": str(session.name),
        "SessionType": session_type,
        "EventDate": str(event["EventDate"]),
    }


def _save_laps(session, out_dir: Path) -> Optional[Path]:
    try:
        if session.laps is None or session.laps.empty:
            return None
    except f1_exceptions.DataNotLoadedError:
        return None

    df = _normalize_laps(session.laps, _session_meta(session))
    out_path = out_dir / "laps.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _save_results(session, out_dir: Path) -> Optional[Path]:
    try:
        if session.results is None or session.results.empty:
            return None
    except f1_exceptions.DataNotLoadedError:
        return None

    df = session.results.copy()
    meta = _session_meta(session)
    for key, value in meta.items():
        df[key] = value

    out_path = out_dir / "results.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _save_drivers(session, out_dir: Path) -> Optional[Path]:
    try:
        results = session.results
    except f1_exceptions.DataNotLoadedError:
        results = None

    drivers_df = None

    if results is not None and not results.empty:
        drivers_df = session.results[
            [
                "DriverNumber",
                "Abbreviation",
                "FullName",
                "TeamName",
            ]
        ].drop_duplicates()
    else:
        try:
            laps = session.laps
        except f1_exceptions.DataNotLoadedError:
            laps = None

    if results is None and laps is not None and not laps.empty:
        columns = [
            col
            for col in ["DriverNumber", "Driver", "Abbreviation", "Team"]
            if col in session.laps.columns
        ]
        if not columns:
            return None
        drivers_df = session.laps[columns].drop_duplicates()
    elif results is None and drivers_df is None:
        return None

    if drivers_df is None:
        return None

    meta = _session_meta(session)
    for key, value in meta.items():
        drivers_df[key] = value

    out_path = out_dir / "drivers.csv"
    drivers_df.to_csv(out_path, index=False)
    return out_path


def _save_weather(session, out_dir: Path) -> Optional[Path]:
    try:
        if session.weather_data is None or session.weather_data.empty:
            event_name = str(session.event["EventName"])
            print(f"[warn] No weather data for {event_name} {session.name}")
            return None
    except f1_exceptions.DataNotLoadedError:
        return None

    df = session.weather_data.copy()
    meta = _session_meta(session)
    for key, value in meta.items():
        df[key] = value

    out_path = out_dir / "weather.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _load_session(year: int, round_number: int, session_code: str):
    session = fastf1.get_session(year, round_number, session_code)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def collect_for_seasons(
    years: Iterable[int],
    session_codes: Iterable[str],
    rounds: Optional[List[int]] = None,
    max_rounds: Optional[int] = None,
) -> None:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _safe_mkdir(RAW_DIR)

    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if rounds:
            schedule = schedule[schedule["RoundNumber"].isin(rounds)]
        if max_rounds:
            schedule = schedule.head(max_rounds)

        for _, row in schedule.iterrows():
            round_number = int(row["RoundNumber"])
            event_name_stripped = str(row["EventName"]).split()
            event_name = "_".join(event_name_stripped)

            for session_code in session_codes:
                try:
                    session = _load_session(year, round_number, session_code)
                except (
                    Exception
                ) as exc:  # FastF1 raises various errors for missing sessions
                    print(
                        f"[skip] {year} {event_name} ({round_number}) {session_code}: {exc}"
                    )
                    continue

                out_dir = (
                    RAW_DIR
                    / str(year)
                    / f"{round_number:02d}_{event_name}"
                    / session_code
                )
                _safe_mkdir(out_dir)

                try:
                    laps_path = _save_laps(session, out_dir)
                    results_path = _save_results(session, out_dir)
                    drivers_path = _save_drivers(session, out_dir)
                    weather_path = _save_weather(session, out_dir)
                except f1_exceptions.DataNotLoadedError as exc:
                    print(
                        f"[skip] {year} {event_name} ({round_number}) {session_code}: "
                        f"data not loaded yet ({exc}). This often happens for current-year sessions."
                    )
                    continue

                print(
                    f"[ok] {year} {event_name} ({round_number}) {session_code} -> "
                    f"laps={bool(laps_path)}, results={bool(results_path)}, drivers={bool(drivers_path)}, weather={bool(weather_path)}"
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: collect raw F1 session data with FastF1 (including sprint weekends)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to download (e.g. 2022 2023)",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"],
        help="Session codes to fetch (default: FP1 FP2 FP3 SQ S Q R)",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        help="Optional list of specific rounds to fetch",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        help="Optional limit on rounds per season (useful for quick tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    collect_for_seasons(
        years=args.years,
        session_codes=args.sessions,
        rounds=args.rounds,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
