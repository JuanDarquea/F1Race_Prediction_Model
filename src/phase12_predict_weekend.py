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
from pathlib import Path
from typing import Tuple

import pandas as pd
from xgboost import XGBClassifier

from common.classifier_utils import align_train_test, clean_and_encode, feature_columns
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

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
WEEKEND_DIR = MODEL_DIR / "weekend"


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
) -> pd.DataFrame:
    """Return one feature row per driver for the target weekend.

    When has_real_data is True, uses the real rows already in the dataset
    for that round. Otherwise builds a historical proxy: the most recent
    round's rows (this year if any exist, else the latest round overall),
    relabeled to the target season/round/track, with elo_pre_race replaced
    by each driver's latest known post-race rating (not the stale rating
    from whichever round the proxy was borrowed from).
    """
    if has_real_data:
        return df_with_elo[
            (df_with_elo["season"] == year) & (df_with_elo["round"] == round_number)
        ].copy()

    season_rows = df_with_elo[df_with_elo["season"] == year]
    if not season_rows.empty:
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

    latest_elo = (
        elo_history.sort_values(["season", "round"])
        .groupby("driver_name")["elo_post_race"]
        .last()
    )
    proxy["elo_pre_race"] = (
        proxy["driver_name"].map(latest_elo).fillna(proxy["elo_pre_race"])
    )
    return proxy


def _fit_final_winner_podium(
    df_with_elo: pd.DataFrame, weekend_raw: pd.DataFrame
) -> pd.DataFrame:
    working = df_with_elo.dropna(subset=["race_position"]).copy()
    working["finish_class"] = _finish_class(working["race_position"])

    train_enc = clean_and_encode(
        working, drop_cols=WINNER_PODIUM_DROP_COLS + ["race_position"]
    )
    weekend_enc = clean_and_encode(
        weekend_raw, drop_cols=WINNER_PODIUM_DROP_COLS + ["race_position"]
    )
    train_enc, weekend_enc = align_train_test(train_enc, weekend_enc)
    cols = feature_columns(train_enc, NON_FEATURE_COLS + ["finish_class"])

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        random_state=42,
    )
    model.fit(train_enc[cols], train_enc["finish_class"])
    classes = list(model.classes_)
    proba = model.predict_proba(weekend_enc[cols])

    win_idx = classes.index(1) if 1 in classes else None
    top3_idx = [classes.index(c) for c in (1, 2, 3) if c in classes]

    out = weekend_raw[["driver_name", "team", "season", "round"]].reset_index(drop=True)
    out = out.copy()
    out["p_win"] = proba[:, win_idx] if win_idx is not None else 0.0
    out["p_top3"] = proba[:, top3_idx].sum(axis=1)
    return out


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


def predict_weekend(
    df_with_elo: pd.DataFrame, weekend_raw: pd.DataFrame
) -> pd.DataFrame:
    """Fit one final model per classifier on all available seasons and
    predict P(win), P(top3), P(top10 race), P(top10 qualifying) for every
    driver in weekend_raw. Returns one combined table sorted by P(win)."""
    out = _fit_final_winner_podium(df_with_elo, weekend_raw)
    out["p_top10_race"] = _fit_final_top10(
        df_with_elo, weekend_raw, "race_position", TOP10_RACE_DROP_COLS, "p_top10_race"
    ).to_numpy()
    out["p_top10_qualifying"] = _fit_final_top10(
        df_with_elo,
        weekend_raw,
        "qualifying_position",
        TOP10_QUALIFYING_DROP_COLS,
        "p_top10_qualifying",
    ).to_numpy()
    return out.sort_values("p_win", ascending=False).reset_index(drop=True)


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
    weekend_raw = _build_weekend_features(
        df_with_elo, elo_history, args.year, round_number, track_name, has_real_data
    )
    if weekend_raw.empty:
        raise ValueError(
            f"No feature rows available to build a proxy for {args.year} round {round_number}."
        )

    predictions = predict_weekend(df_with_elo, weekend_raw)
    predictions.insert(2, "track", track_name)

    out_path = WEEKEND_DIR / f"predict_{args.year}_round{round_number:02d}.csv"
    predictions.to_csv(out_path, index=False)

    report_path = WEEKEND_DIR / f"report_{args.year}_round{round_number:02d}.txt"
    with report_path.open("w") as f:
        f.write("Phase 12 - Live Weekend Prediction\n")
        f.write("===================================\n\n")
        f.write(f"{track_name} - {args.year} Round {round_number}\n")
        f.write(f"Real session data available for this round: {has_real_data}\n")
        if not has_real_data:
            f.write(
                "This round has not happened yet (or has no data collected) - "
                "features are a historical proxy using each driver's most recent "
                "known form, with Elo refreshed to their latest known rating. "
                "Re-run after qualifying/race data is collected for a more "
                "accurate result.\n"
            )
        f.write("\nPredictions (sorted by P(win)):\n\n")
        f.write(predictions.to_string(index=False))
        f.write("\n")

    print(f"Saved predictions -> {out_path}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
