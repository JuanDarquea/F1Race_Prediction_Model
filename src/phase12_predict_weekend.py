"""Phase 12: Live weekend prediction using the Phase 9 Elo ratings and the
Phase 10 classifier logic.

Phase 10's `train_*` functions are a walk-forward EVALUATION harness: each
fold retrains a model purely to score it against a season that's already
happened, then discards the model. This script is different — it fits one
final model per classifier on every available season and predicts a single
requested race weekend, including a genuinely future round (no session data
collected yet) via a historical-proxy fallback: the driver's most recent
known feature row, relabeled to the target round, with Elo refreshed to
each driver's latest known rating. This mirrors how `phase6_predict_2026.py`
already handles an unrun round for the regression models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from common.classifier_utils import align_train_test, clean_and_encode, feature_columns
from common.walk_forward import expanding_season_folds
from phase4_feature_engineering import (
    _load_practice_pace,
    add_practice_rank_features,
)
from phase10_classifiers import (
    ELO_HISTORY_PATH,
    FEATURE_PATH,
    NON_FEATURE_COLS,
    TOP10_QUALIFYING_DROP_COLS,
    TOP10_RACE_DROP_COLS,
    WINNER_PODIUM_DROP_COLS,
    _finish_class,
    _fit_calibrated_binary,
    load_dataset_with_elo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_TYPE_PATH = PROJECT_ROOT / "data" / "track_types.csv"
MODEL_DIR = PROJECT_ROOT / "models"
WEEKEND_DIR = MODEL_DIR / "weekend"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fastf1"

WIN_TEMPERATURES = np.arange(1.0, 6.01, 0.25)
PODIUM_SPOTS = 3


def load_qualifying_grid(
    year: int, round_number: int, raw_root: Path = RAW_ROOT
) -> Optional[Dict[str, float]]:
    """Real qualifying result for a round as {driver_name: position}, or None
    if the session hasn't happened / has no classified results yet."""
    for round_dir in sorted((raw_root / str(year)).glob(f"{round_number:02d}_*")):
        results_path = round_dir / "Q" / "results.csv"
        if not results_path.exists():
            continue
        results = pd.read_csv(results_path, usecols=["FullName", "Position"])
        results = results.dropna(subset=["FullName", "Position"])
        if not results.empty:
            return dict(zip(results["FullName"], results["Position"].astype(float)))
    return None


def load_weekend_roster(
    year: int, round_number: int, raw_root: Path = RAW_ROOT
) -> Optional[Dict[str, str]]:
    """Who is actually racing this weekend, as {driver_name: team}. Read from the
    latest session already run (Q, then FP3..FP1), or None if nothing has run."""
    for round_dir in sorted((raw_root / str(year)).glob(f"{round_number:02d}_*")):
        for session in ("Q", "FP3", "FP2", "FP1"):
            results_path = round_dir / session / "results.csv"
            if not results_path.exists():
                continue
            results = pd.read_csv(results_path, usecols=["FullName", "TeamName"])
            results = results.dropna(subset=["FullName", "TeamName"])
            if not results.empty:
                return dict(zip(results["FullName"], results["TeamName"]))
    return None


def load_practice_pace(
    year: int, round_number: int, raw_root: Path = RAW_ROOT
) -> Optional[Dict[str, float]]:
    """Best practice lap (seconds) per driver over the round's FP sessions run
    so far, as {driver_name: seconds}, or None if no practice data exists yet."""
    pace = _load_practice_pace(raw_root / str(year))
    pace = pace[pace["round"] == round_number].dropna(subset=["practice_pace"])
    if pace.empty:
        return None
    return dict(zip(pace["driver_name"], pace["practice_pace"].astype(float)))


def load_track_type(track_name: str, path: Path = TRACK_TYPE_PATH) -> Optional[str]:
    """'street' / 'race' label for a circuit from data/track_types.csv, if listed."""
    if not path.exists():
        return None
    table = pd.read_csv(path)
    match = table[table["track"] == track_name]
    return str(match["track_type"].iloc[0]) if not match.empty else None


TEAM_LEVEL_COLS = [
    "team_avg_finish",
    "team_avg_qualifying",
    "constructor_points",
    "team_points_last_5",
    "team_dnf_rate_last_10",
    "team_avg_pit_time_last_5",
]


def _roster_rows(history: pd.DataFrame, roster: Dict[str, str]) -> pd.DataFrame:
    """One row per driver on the real roster: their own latest known row, moved
    to the team they drive for now (team-level columns follow the car). Drivers
    with no history at all can't be featurized and are left out."""
    ordered = history.sort_values(["season", "round"])
    rows = ordered[ordered["driver_name"].isin(roster)].groupby("driver_name").tail(1)
    rows = rows.copy()
    rows["team"] = rows["driver_name"].map(roster)
    latest_by_team = ordered.groupby("team").tail(1).set_index("team")
    for col in TEAM_LEVEL_COLS:
        if col in rows.columns:
            follow_car = rows["team"].map(latest_by_team[col])
            rows[col] = follow_car.where(follow_car.notna(), rows[col])
    return rows


def _refresh_track_features(
    proxy: pd.DataFrame,
    history: pd.DataFrame,
    track_name: str,
    track_type: Optional[str],
) -> pd.DataFrame:
    """A proxy row is borrowed from the previous circuit, so its circuit-specific
    columns describe the wrong track. Recompute them for the target circuit from
    every earlier race there: mean finish per driver / per team, plus its type.
    A circuit with no history leaves them missing, as in training for new tracks."""
    past = history[history["track"] == track_name]
    proxy["driver_performance_at_track"] = proxy["driver_name"].map(
        past.groupby("driver_name")["race_position"].mean()
    )
    proxy["team_performance_at_track"] = proxy["team"].map(
        past.groupby("team")["race_position"].mean()
    )
    if track_type is not None:
        proxy["track_type"] = track_type
    return proxy


def _lookup_track_name(year: int, round_number: int) -> str:
    """Best-effort track name for a round with no data yet. Never raises —
    falls back to a generic label if the FastF1 schedule call fails for any
    reason (no network, unknown round, etc.)."""
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        match = schedule[schedule["RoundNumber"] == round_number]
        if not match.empty:
            return str(match.iloc[0]["EventName"])
    except Exception:
        pass
    return f"Round {round_number}"


def _select_weekend(
    df: pd.DataFrame, year: int, round_number: int | None
) -> Tuple[int, str, bool]:
    """Return (round_number, track_name, has_real_data).

    If round_number is omitted, targets the NEXT round after the latest one
    already in the data for that year (i.e. the upcoming, not-yet-run race) —
    unlike phase6's default of the latest round WITH data, which is a race
    that already happened.
    """
    season_rows = (
        df[df["season"] == year][["round", "track"]]
        .drop_duplicates()
        .sort_values("round")
    )

    if round_number is None:
        round_number = (
            int(season_rows["round"].max()) + 1 if not season_rows.empty else 1
        )

    match = season_rows[season_rows["round"] == round_number]
    if not match.empty:
        return int(round_number), str(match.iloc[0]["track"]), True

    return int(round_number), _lookup_track_name(year, round_number), False


def _build_weekend_features(
    df_with_elo: pd.DataFrame,
    elo_history: pd.DataFrame,
    year: int,
    round_number: int,
    track_name: str,
    has_real_data: bool,
    qualifying_grid: Optional[Dict[str, float]] = None,
    practice_pace: Optional[Dict[str, float]] = None,
    track_type: Optional[str] = None,
    roster: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Return one feature row per driver for the target weekend.

    When has_real_data is True, uses the real rows already in the dataset
    for that round. Otherwise builds a historical proxy: the most recent
    round's rows (this year if any exist, else the latest round overall),
    relabeled to the target season/round/track, with elo_pre_race replaced
    by each driver's latest known post-race rating (not the stale rating
    from whichever round the proxy was borrowed from). If qualifying_grid
    is given (qualifying done, race not yet), the proxy's qualifying_position
    is replaced with the real grid; drivers missing from it are placed behind
    everyone who set a time. If practice_pace is given, the proxy's practice
    columns are rebuilt from this weekend's real practice laps (a driver with
    no lap gets a missing value, imputed downstream like any other gap). The
    circuit-specific columns are always recomputed for the target circuit.
    If roster ({driver: team}) is given, the proxy's driver list is replaced by
    it, so lineup changes since the last round (a driver swap, a return) carry
    over instead of copying the previous round's lineup.
    """
    if has_real_data:
        return df_with_elo[
            (df_with_elo["season"] == year) & (df_with_elo["round"] == round_number)
        ].copy()

    season_rows = df_with_elo[df_with_elo["season"] == year]
    if roster:
        proxy = _roster_rows(df_with_elo, roster)
    elif not season_rows.empty:
        proxy_round = int(season_rows["round"].max())
        proxy = season_rows[season_rows["round"] == proxy_round].copy()
    else:
        proxy_round = int(df_with_elo["round"].max())
        latest_season = int(
            df_with_elo[df_with_elo["round"] == proxy_round]["season"].max()
        )
        proxy = df_with_elo[
            (df_with_elo["season"] == latest_season)
            & (df_with_elo["round"] == proxy_round)
        ].copy()

    proxy["season"] = year
    proxy["round"] = round_number
    proxy["track"] = track_name
    proxy = _refresh_track_features(proxy, df_with_elo, track_name, track_type)

    latest_elo = (
        elo_history.sort_values(["season", "round"])
        .groupby("driver_name")["elo_post_race"]
        .last()
    )
    proxy["elo_pre_race"] = (
        proxy["driver_name"].map(latest_elo).fillna(proxy["elo_pre_race"])
    )
    if practice_pace:
        proxy["practice_pace"] = proxy["driver_name"].map(practice_pace)
        proxy = add_practice_rank_features(proxy)
    if qualifying_grid:
        last_place = max(qualifying_grid.values())
        proxy["qualifying_position"] = proxy["driver_name"].map(qualifying_grid)
        missing = proxy["qualifying_position"].isna()
        proxy.loc[missing, "qualifying_position"] = last_place + 1
    return proxy


def _winner_matrices(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    working = train_df.dropna(subset=["race_position"]).copy()
    working["finish_class"] = _finish_class(working["race_position"])
    drop_cols = WINNER_PODIUM_DROP_COLS + ["race_position"]

    train_enc = clean_and_encode(working, drop_cols=drop_cols)
    test_enc = clean_and_encode(test_df, drop_cols=drop_cols)
    train_enc, test_enc = align_train_test(train_enc, test_enc)
    cols = feature_columns(train_enc, NON_FEATURE_COLS + ["finish_class"])
    return train_enc[cols], train_enc["finish_class"], test_enc[cols]


def _winner_probabilities(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw (uncalibrated) per-driver P(win) and P(top3) from one softmax model."""
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        random_state=42,
    )
    model.fit(X_train, y_train)
    classes = list(model.classes_)
    proba = model.predict_proba(X_test)
    p_win = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X_test))
    top3_idx = [classes.index(c) for c in (1, 2, 3) if c in classes]
    return p_win, proba[:, top3_idx].sum(axis=1)


def _fit_final_winner_podium(
    df_with_elo: pd.DataFrame, weekend_raw: pd.DataFrame
) -> pd.DataFrame:
    X_train, y_train, X_weekend = _winner_matrices(df_with_elo, weekend_raw)
    p_win, p_top3 = _winner_probabilities(X_train, y_train, X_weekend)

    out = weekend_raw[["driver_name", "team", "season", "round"]].reset_index(drop=True)
    out = out.copy()
    out["p_win"] = p_win
    out["p_top3"] = p_top3
    return out


def oof_winner_podium(df_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward out-of-fold raw P(win)/P(top3) for every finished race in
    every test season, with the actual race_position, for fitting calibrators."""
    finished = df_with_elo.dropna(subset=["race_position"])
    frames = []
    for train_seasons, test_season in expanding_season_folds(finished):
        train_df = finished[finished["season"].isin(train_seasons)]
        test_df = finished[finished["season"] == test_season]
        X_train, y_train, X_test = _winner_matrices(train_df, test_df)
        p_win, p_top3 = _winner_probabilities(X_train, y_train, X_test)
        frame = test_df[["season", "round", "driver_name", "race_position"]].copy()
        frame["p_win"] = p_win
        frame["p_top3"] = p_top3
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "driver_name",
                "race_position",
                "p_win",
                "p_top3",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def temper_win_probabilities(p_win: np.ndarray, temperature: float) -> np.ndarray:
    """Turn raw per-driver P(win) into one distribution over the field that sums
    to 1: p^(1/T), renormalized. T > 1 flattens an overconfident model."""
    scores = np.clip(np.asarray(p_win, dtype=float), 1e-9, None) ** (1.0 / temperature)
    return scores / scores.sum()


def fit_win_temperature(
    oof: pd.DataFrame, prob_col: str = "p_win", position_col: str = "race_position"
) -> float:
    """Temperature minimizing the mean race-level log-loss of the actual
    winner's (first place's) tempered probability over the out-of-fold races."""
    races = [
        (r[prob_col].to_numpy(), (r[position_col] == 1).to_numpy())
        for _, r in oof.groupby(["season", "round"])
        if (r[position_col] == 1).any()
    ]
    if not races:
        return 1.0

    def loss(temperature: float) -> float:
        return float(
            np.mean(
                [
                    -np.log(max(temper_win_probabilities(p, temperature)[win][0], 1e-9))
                    for p, win in races
                ]
            )
        )

    return float(min(WIN_TEMPERATURES, key=loss))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def fit_podium_calibrator(
    oof: pd.DataFrame,
    prob_col: str = "p_top3",
    position_col: str = "race_position",
    spots: int = PODIUM_SPOTS,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Platt scaling (logistic on logit(raw P(top3))) fit on out-of-fold races.
    Two parameters, so it can't overfit the thin high-probability tail the way
    isotonic regression did; it also scored better leave-one-season-out."""
    if oof.empty:
        return None
    y = (oof[position_col] <= spots).astype(int)
    if y.nunique() < 2:
        return None
    model = LogisticRegression(C=1e6).fit(_logit(oof[prob_col]).reshape(-1, 1), y)
    return lambda raw: model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


def normalize_podium_probabilities(
    p: np.ndarray, total: float = PODIUM_SPOTS
) -> np.ndarray:
    """Shift every driver's logit by one constant so the field sums to `total`
    (there are only three podium spots). Bisection; probabilities stay in (0, 1)."""
    p = np.asarray(p, dtype=float)
    if len(p) <= total:
        return p
    base = _logit(p)
    low, high = -20.0, 20.0
    for _ in range(60):
        mid = (low + high) / 2
        if (1 / (1 + np.exp(-(base + mid)))).sum() > total:
            high = mid
        else:
            low = mid
    return 1 / (1 + np.exp(-(base + (low + high) / 2)))


def _fit_final_top10(
    df_with_elo: pd.DataFrame,
    weekend_raw: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    prob_col: str,
) -> pd.Series:
    working = df_with_elo.dropna(subset=[target_col]).copy()
    working["is_top10"] = (working[target_col] <= 10).astype(int)

    train_enc = clean_and_encode(working, drop_cols=drop_cols + [target_col])
    weekend_enc = clean_and_encode(weekend_raw, drop_cols=drop_cols + [target_col])
    train_enc, weekend_enc = align_train_test(train_enc, weekend_enc)
    cols = feature_columns(train_enc, NON_FEATURE_COLS + ["is_top10"])

    y_prob = _fit_calibrated_binary(
        train_enc[cols], train_enc["is_top10"], weekend_enc[cols]
    )
    return pd.Series(y_prob, name=prob_col)


QUALI_SLOTS = 5


def _quali_matrices(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    working = train_df.dropna(subset=["qualifying_position"]).copy()
    working["quali_class"] = np.where(
        working["qualifying_position"] <= QUALI_SLOTS, working["qualifying_position"], 0
    ).astype(int)
    drop_cols = TOP10_QUALIFYING_DROP_COLS + ["qualifying_position"]

    train_enc = clean_and_encode(working, drop_cols=drop_cols)
    test_enc = clean_and_encode(test_df, drop_cols=drop_cols)
    train_enc, test_enc = align_train_test(train_enc, test_enc)
    cols = feature_columns(train_enc, NON_FEATURE_COLS + ["quali_class"])
    return train_enc[cols], train_enc["quali_class"], test_enc[cols]


def _quali_probabilities(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> pd.DataFrame:
    """Raw per-driver P(pole), P(top 3 on the grid), P(top 5) from one softmax
    over grid slots 1-5 plus 'sixth or worse'."""
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        random_state=42,
    )
    model.fit(X_train, y_train)
    classes = list(model.classes_)
    proba = model.predict_proba(X_test)

    def cumulative(k: int) -> np.ndarray:
        idx = [classes.index(c) for c in range(1, k + 1) if c in classes]
        return proba[:, idx].sum(axis=1) if idx else np.zeros(len(X_test))

    return pd.DataFrame(
        {
            "p_pole": cumulative(1),
            "p_quali_top3": cumulative(3),
            "p_quali_top5": cumulative(5),
        }
    )


def oof_qualifying(df_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward out-of-fold raw grid probabilities with the actual
    qualifying_position, for fitting the qualifying calibrators."""
    qualified = df_with_elo.dropna(subset=["qualifying_position"])
    frames = []
    for train_seasons, test_season in expanding_season_folds(qualified):
        train_df = qualified[qualified["season"].isin(train_seasons)]
        test_df = qualified[qualified["season"] == test_season]
        X_train, y_train, X_test = _quali_matrices(train_df, test_df)
        keep = ["season", "round", "driver_name", "qualifying_position"]
        if "practice_pace_rank" in test_df.columns:
            keep.append("practice_pace_rank")
        frame = test_df[keep].reset_index(drop=True)
        frames.append(
            pd.concat([frame, _quali_probabilities(X_train, y_train, X_test)], axis=1)
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "driver_name",
                "qualifying_position",
                "p_pole",
                "p_quali_top3",
                "p_quali_top5",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def predicted_grid_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted grid, slot by slot, so each slot is ranked by the probability
    that slot is about: pole = highest P(pole); slots 2-3 = highest P(top 3) of
    the rest; slots 4-5 = highest P(top 5) of the rest; the remainder by P(top
    10 qualifying) when available. Adds `grid_slot` (1 = pole), keeps the index."""
    remaining = frame.copy()
    picks = []
    for slots, col in ((1, "p_pole"), (2, "p_quali_top3"), (2, "p_quali_top5")):
        chosen = remaining.sort_values(col, ascending=False, kind="stable").head(slots)
        picks.append(chosen)
        remaining = remaining.drop(chosen.index)
    tail = "p_top10_qualifying" if "p_top10_qualifying" in remaining else "p_quali_top5"
    picks.append(remaining.sort_values(tail, ascending=False, kind="stable"))
    order = pd.concat(picks)
    order["grid_slot"] = range(1, len(order) + 1)
    return order


def calibrate_qualifying(
    out: pd.DataFrame, oof: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """Calibrate the grid probabilities out-of-fold (raw kept as *_raw): P(pole)
    is tempered into one distribution over the field; top-3 / top-5 go through
    Platt scaling and are normalized to 3 / 5 slots, then forced to be nested
    (pole <= top 3 <= top 5). Without out-of-fold data the raw values are kept."""
    out = out.copy()
    for col in ("p_pole", "p_quali_top3", "p_quali_top5"):
        out[f"{col}_raw"] = out[col]
    if oof.empty:
        return out, 1.0

    temperature = fit_win_temperature(oof, "p_pole", "qualifying_position")
    out["p_pole"] = temper_win_probabilities(out["p_pole_raw"].to_numpy(), temperature)
    for col, spots in (("p_quali_top3", 3), ("p_quali_top5", QUALI_SLOTS)):
        calibrator = fit_podium_calibrator(oof, col, "qualifying_position", spots)
        if calibrator is not None:
            out[col] = normalize_podium_probabilities(
                calibrator(out[f"{col}_raw"].to_numpy()), total=spots
            )
    out["p_quali_top3"] = np.maximum(out["p_quali_top3"], out["p_pole"])
    out["p_quali_top5"] = np.maximum(out["p_quali_top5"], out["p_quali_top3"])
    return out, temperature


def summarize_quali_oof(oof: pd.DataFrame) -> Dict[str, float]:
    """Honest out-of-fold record for the predicted grid (same slot-by-slot rule
    as the published picks): how often the pole pick took pole, and the share of
    the real top 3 / top 5 recovered."""
    races = [
        r
        for _, r in oof.groupby(["season", "round"])
        if (r["qualifying_position"] == 1).any()
    ]
    if not races:
        return {}
    pole: List[bool] = []
    top3: List[float] = []
    top5: List[float] = []
    for r in races:
        ranked = predicted_grid_order(r)
        pole.append(ranked["qualifying_position"].iloc[0] == 1)
        for spots, bucket in ((3, top3), (QUALI_SLOTS, top5)):
            real = set(r[r["qualifying_position"] <= spots].index)
            bucket.append(len(set(ranked.head(spots).index) & real) / spots)
    summary = {
        "n_sessions": len(races),
        "first_season": int(oof["season"].min()),
        "last_season": int(oof["season"].max()),
        "pole_accuracy": float(np.mean(pole)),
        "top3_overlap": float(np.mean(top3)),
        "top5_overlap": float(np.mean(top5)),
    }
    if "practice_pace_rank" in oof.columns:
        leader_took_pole = [
            r.loc[r["practice_pace_rank"] == 1, "qualifying_position"].iloc[0] == 1
            for r in races
            if (r["practice_pace_rank"] == 1).any()
        ]
        if leader_took_pole:
            summary["practice_leader_pole_rate"] = float(np.mean(leader_took_pole))
    return summary


def summarize_oof(oof: pd.DataFrame) -> Dict[str, float]:
    """Honest out-of-fold track record: how often the top P(win) pick won, and
    how much of the real podium the top-3 by P(top3) covered (0-1, per race)."""
    races = [
        r
        for _, r in oof.groupby(["season", "round"])
        if (r["race_position"] == 1).any()
    ]
    if not races:
        return {}
    winner_hits = [r.loc[r["p_win"].idxmax(), "race_position"] == 1 for r in races]
    overlaps = [
        len(
            set(r.nlargest(PODIUM_SPOTS, "p_top3").index)
            & set(r[r["race_position"] <= PODIUM_SPOTS].index)
        )
        / PODIUM_SPOTS
        for r in races
    ]
    return {
        "n_races": len(races),
        "first_season": int(oof["season"].min()),
        "last_season": int(oof["season"].max()),
        "winner_top1_accuracy": float(np.mean(winner_hits)),
        "podium_overlap": float(np.mean(overlaps)),
    }


def calibrate_winner_podium(
    out: pd.DataFrame, oof: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """Replace raw p_win/p_top3 with calibrated ones (raw kept as *_raw).

    p_win becomes one tempered distribution over the field (sums to 1) using
    the temperature that minimizes out-of-fold winner log-loss; p_top3 goes
    through a Platt calibrator fit on the same races, then is normalized so
    the field sums to three podium spots.
    With no out-of-fold data the raw values are kept and temperature is 1.0.
    """
    out = out.copy()
    out["p_win_raw"] = out["p_win"]
    out["p_top3_raw"] = out["p_top3"]
    if oof.empty:
        return out, 1.0

    temperature = fit_win_temperature(oof)
    out["p_win"] = temper_win_probabilities(out["p_win_raw"].to_numpy(), temperature)

    calibrator = fit_podium_calibrator(oof)
    if calibrator is not None:
        calibrated = calibrator(out["p_top3_raw"].to_numpy())
        out["p_top3"] = normalize_podium_probabilities(calibrated)
    return out, temperature


def _fit_final_quali(
    df_with_elo: pd.DataFrame, weekend_raw: pd.DataFrame
) -> pd.DataFrame:
    X_train, y_train, X_weekend = _quali_matrices(df_with_elo, weekend_raw)
    return _quali_probabilities(X_train, y_train, X_weekend)


def predict_weekend(
    df_with_elo: pd.DataFrame,
    weekend_raw: pd.DataFrame,
    calibrate: bool = True,
    use_predicted_grid: bool = False,
) -> pd.DataFrame:
    """Fit one final model per classifier on all available seasons and predict,
    for every driver in weekend_raw: P(win), P(top3), P(top10 race), P(top10
    qualifying) and the grid odds P(pole), P(top 3 grid), P(top 5 grid). Returns
    one combined table sorted by P(win); temperatures and out-of-fold track
    records are stored in `.attrs`.

    With use_predicted_grid (qualifying not run yet) the race models are fed our
    own predicted grid instead of the stale qualifying_position of the borrowed
    proxy row, and the table gains `predicted_grid_slot`."""
    quali = _fit_final_quali(df_with_elo, weekend_raw)
    out = weekend_raw[["driver_name", "team", "season", "round"]].reset_index(drop=True)
    for col in quali.columns:
        out[col] = quali[col].to_numpy()
    quali_temperature = 1.0
    quali_summary: Dict[str, float] = {}
    if calibrate:
        oof_q = oof_qualifying(df_with_elo)
        out, quali_temperature = calibrate_qualifying(out, oof_q)
        quali_summary = summarize_quali_oof(oof_q)

    race_input = weekend_raw
    if use_predicted_grid:
        slots = out.assign(
            p_top10_qualifying=_fit_final_top10(
                df_with_elo,
                weekend_raw,
                "qualifying_position",
                TOP10_QUALIFYING_DROP_COLS,
                "p_top10_qualifying",
            ).to_numpy()
        )
        order = predicted_grid_order(slots)
        slot_by_driver = dict(
            zip(order["driver_name"], order["grid_slot"].astype(float))
        )
        race_input = weekend_raw.copy()
        race_input["qualifying_position"] = race_input["driver_name"].map(
            slot_by_driver
        )
        out["predicted_grid_slot"] = out["driver_name"].map(slot_by_driver)

    race = _fit_final_winner_podium(df_with_elo, race_input)
    for col in ("p_win", "p_top3"):
        out[col] = race[col].to_numpy()
    temperature = 1.0
    summary: Dict[str, float] = {}
    if calibrate:
        oof = oof_winner_podium(df_with_elo)
        race_cal, temperature = calibrate_winner_podium(race, oof)
        for col in ("p_win", "p_top3", "p_win_raw", "p_top3_raw"):
            out[col] = race_cal[col].to_numpy()
        summary = summarize_oof(oof)
    out["p_top10_race"] = _fit_final_top10(
        df_with_elo, race_input, "race_position", TOP10_RACE_DROP_COLS, "p_top10_race"
    ).to_numpy()
    out["p_top10_qualifying"] = _fit_final_top10(
        df_with_elo,
        weekend_raw,
        "qualifying_position",
        TOP10_QUALIFYING_DROP_COLS,
        "p_top10_qualifying",
    ).to_numpy()
    out = out.sort_values(["p_win", "p_top3"], ascending=False).reset_index(drop=True)
    out.attrs["win_temperature"] = temperature
    out.attrs["oof_summary"] = summary
    out.attrs["pole_temperature"] = quali_temperature
    out.attrs["quali_summary"] = quali_summary
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 12: Live weekend prediction (winner/podium + top-10 race/qualifying)"
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--elo-history", default=str(ELO_HISTORY_PATH))
    args = parser.parse_args()

    WEEKEND_DIR.mkdir(parents=True, exist_ok=True)

    df_with_elo = load_dataset_with_elo(Path(args.features), Path(args.elo_history))
    elo_history = pd.read_csv(args.elo_history)

    round_number, track_name, has_real_data = _select_weekend(
        df_with_elo, args.year, args.round
    )
    qualifying_grid = (
        None if has_real_data else load_qualifying_grid(args.year, round_number)
    )
    practice_pace = (
        None if has_real_data else load_practice_pace(args.year, round_number)
    )
    roster = None if has_real_data else load_weekend_roster(args.year, round_number)
    weekend_raw = _build_weekend_features(
        df_with_elo,
        elo_history,
        args.year,
        round_number,
        track_name,
        has_real_data,
        qualifying_grid,
        practice_pace,
        load_track_type(track_name),
        roster,
    )
    if weekend_raw.empty:
        raise ValueError(
            f"No feature rows available to build a proxy for {args.year} round {round_number}."
        )

    predicted_grid = not has_real_data and qualifying_grid is None
    predictions = predict_weekend(
        df_with_elo, weekend_raw, use_predicted_grid=predicted_grid
    )
    temperature = predictions.attrs.get("win_temperature", 1.0)
    predictions.insert(2, "track", track_name)

    stem = f"{args.year}_round{round_number:02d}"
    out_path = WEEKEND_DIR / f"predict_{stem}.csv"
    predictions.to_csv(out_path, index=False)

    meta = {
        "year": args.year,
        "round": round_number,
        "track": track_name,
        "has_real_data": has_real_data,
        "has_qualifying_grid": qualifying_grid is not None,
        "has_practice_data": practice_pace is not None,
        "grid_source": (
            "actual" if has_real_data else "real" if qualifying_grid else "predicted"
        ),
        "win_temperature": temperature,
        "validation": predictions.attrs.get("oof_summary", {}),
        "pole_temperature": predictions.attrs.get("pole_temperature", 1.0),
        "validation_quali": predictions.attrs.get("quali_summary", {}),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (WEEKEND_DIR / f"meta_{stem}.json").write_text(json.dumps(meta, indent=2))

    report_path = WEEKEND_DIR / f"report_{stem}.txt"
    with report_path.open("w") as f:
        f.write("Phase 12 - Live Weekend Prediction\n")
        f.write("===================================\n\n")
        f.write(f"{track_name} - {args.year} Round {round_number}\n")
        f.write(f"Real session data available for this round: {has_real_data}\n")
        f.write(f"Real qualifying grid used: {qualifying_grid is not None}\n")
        f.write(f"Real practice pace used: {practice_pace is not None}\n")
        f.write(f"P(win) temperature (out-of-fold fit): {temperature:.2f}\n")
        if not has_real_data:
            f.write(
                "This round has not happened yet (or has no data collected) - "
                "features are a historical proxy using each driver's most recent "
                "known form, with Elo refreshed to their latest known rating"
                + (
                    ", and the real qualifying grid."
                    if qualifying_grid is not None
                    else ". Re-run after qualifying to use the real grid."
                )
                + "\n"
            )
        f.write("\nPredictions (sorted by P(win)):\n\n")
        f.write(predictions.to_string(index=False))
        f.write("\n")

    print(f"Saved predictions -> {out_path}")
    print(f"Saved report -> {report_path}")
    print(
        f"Win-probability temperature: {temperature:.2f}; "
        f"real grid: {qualifying_grid is not None}; real practice: {practice_pace is not None}"
    )


if __name__ == "__main__":
    main()
