import pandas as pd
import pytest

from phase12_predict_weekend import (
    _build_weekend_features,
    _select_weekend,
    predict_weekend,
)


def _synthetic_dataset() -> pd.DataFrame:
    rows = []
    for season in (2023, 2024, 2025):
        for round_ in (1, 2):
            for driver_idx in range(12):
                position = driver_idx + 1
                rows.append(
                    {
                        "season": season,
                        "round": round_,
                        "track": f"Track{round_}",
                        "driver_name": f"D{driver_idx}",
                        "driver_id": driver_idx + 1,
                        "team": f"Team{driver_idx % 4}",
                        "status": "Finished",
                        "race_position": position,
                        "qualifying_position": position,
                        "race_points": 0,
                        "sprint_position": None,
                        "sprint_points": None,
                        "sprint_qualifying_position": None,
                        "elo_pre_race": 1500.0 + driver_idx * 5 + season - 2023,
                        "practice_pace": 90.0 + driver_idx,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_elo_history() -> pd.DataFrame:
    rows = []
    for driver_idx in range(12):
        rows.append(
            {
                "season": 2025,
                "round": 2,
                "driver_name": f"D{driver_idx}",
                "elo_pre_race": 1500.0,
                "elo_post_race": 1600.0 + driver_idx,
            }
        )
    return pd.DataFrame(rows)


def test_select_weekend_returns_existing_round_when_present():
    df = _synthetic_dataset()
    round_number, track, has_real_data = _select_weekend(df, 2025, 2)
    assert round_number == 2
    assert track == "Track2"
    assert has_real_data is True


def test_select_weekend_auto_increments_past_latest_round(monkeypatch):
    monkeypatch.setattr(
        "phase12_predict_weekend._lookup_track_name",
        lambda year, round_number: "Stub Track",
    )
    df = _synthetic_dataset()
    round_number, track, has_real_data = _select_weekend(df, 2025, None)
    assert round_number == 3
    assert has_real_data is False
    assert track == "Stub Track"


def test_build_weekend_features_uses_real_rows_when_available():
    df = _synthetic_dataset()
    elo_history = _synthetic_elo_history()
    weekend = _build_weekend_features(df, elo_history, 2025, 2, "Track2", True)
    assert set(weekend["driver_name"]) == {f"D{i}" for i in range(12)}
    assert (weekend["season"] == 2025).all()
    assert (weekend["round"] == 2).all()


def test_build_weekend_features_builds_proxy_with_overridden_metadata_and_refreshed_elo():
    df = _synthetic_dataset()
    elo_history = _synthetic_elo_history()
    weekend = _build_weekend_features(df, elo_history, 2025, 3, "Baku", False)

    assert (weekend["season"] == 2025).all()
    assert (weekend["round"] == 3).all()
    assert (weekend["track"] == "Baku").all()
    assert set(weekend["driver_name"]) == {f"D{i}" for i in range(12)}

    d0_elo = weekend.loc[weekend["driver_name"] == "D0", "elo_pre_race"].iloc[0]
    assert d0_elo == pytest.approx(1600.0)


def test_predict_weekend_returns_expected_columns_for_every_driver():
    df = _synthetic_dataset()
    weekend = df[(df["season"] == 2025) & (df["round"] == 2)].copy()

    predictions = predict_weekend(df, weekend)

    assert len(predictions) == 12
    expected_cols = {
        "driver_name",
        "team",
        "season",
        "round",
        "p_win",
        "p_top3",
        "p_top10_race",
        "p_top10_qualifying",
    }
    assert expected_cols.issubset(predictions.columns)
    for col in ("p_win", "p_top3", "p_top10_race", "p_top10_qualifying"):
        assert predictions[col].between(0.0, 1.0).all()
