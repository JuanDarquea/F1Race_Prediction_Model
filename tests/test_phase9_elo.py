import pandas as pd
import pytest

from phase9_elo import (
    DEFAULT_K,
    apply_season_decay,
    build_elo_history,
    expected_score,
    pairwise_race_update,
)


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_rating_favored():
    assert expected_score(1600, 1400) > 0.5


def test_pairwise_race_update_three_equal_drivers():
    pre = {"A": 1500.0, "B": 1500.0, "C": 1500.0}
    positions = {"A": 1, "B": 2, "C": 3}
    post = pairwise_race_update(pre, positions, k=DEFAULT_K)
    assert post["A"] == pytest.approx(1512.0)
    assert post["B"] == pytest.approx(1500.0)
    assert post["C"] == pytest.approx(1488.0)


def test_pairwise_race_update_single_driver_is_noop():
    pre = {"A": 1500.0}
    assert pairwise_race_update(pre, {"A": 1}, k=DEFAULT_K) == pre


def test_apply_season_decay_regresses_toward_mean():
    ratings = {"A": 1600.0, "B": 1500.0, "C": 1400.0}
    decayed = apply_season_decay(ratings, decay=0.75)
    assert decayed["A"] == pytest.approx(1575.0)
    assert decayed["B"] == pytest.approx(1500.0)
    assert decayed["C"] == pytest.approx(1425.0)


def test_build_elo_history_has_expected_columns():
    df = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "driver_name": ["A", "B"],
            "race_position": [1, 2],
        }
    )
    history = build_elo_history(df)
    assert list(history.columns) == [
        "season",
        "round",
        "driver_name",
        "elo_pre_race",
        "elo_post_race",
    ]


def test_build_elo_history_new_driver_starts_at_default_rating():
    df = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "driver_name": ["A", "B"],
            "race_position": [1, 2],
        }
    )
    history = build_elo_history(df, start_rating=1500.0)
    by_driver = history.set_index("driver_name")
    assert by_driver.loc["A", "elo_pre_race"] == pytest.approx(1500.0)
    assert by_driver.loc["A", "elo_post_race"] == pytest.approx(1512.0)
    assert by_driver.loc["B", "elo_post_race"] == pytest.approx(1488.0)


def test_build_elo_history_applies_season_decay_at_boundary():
    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "round": [1, 1, 1, 1],
            "driver_name": ["A", "B", "A", "B"],
            "race_position": [1, 2, 1, 2],
        }
    )
    history = build_elo_history(df, k=24.0, decay=0.75, start_rating=1500.0)
    season_2024 = history[history["season"] == 2024].set_index("driver_name")
    # 2023 leaves A at 1512, B at 1488 (same math as the single-race test above).
    # Season decay toward mean 1500 by 0.75 before 2024 round 1:
    # A: 1500 + 0.75*(1512-1500) = 1509 ; B: 1500 + 0.75*(1488-1500) = 1491
    assert season_2024.loc["A", "elo_pre_race"] == pytest.approx(1509.0)
    assert season_2024.loc["B", "elo_pre_race"] == pytest.approx(1491.0)


def test_build_elo_history_drops_rows_with_missing_race_position():
    df = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "driver_name": ["A", "B"],
            "race_position": [1, None],
        }
    )
    history = build_elo_history(df)
    assert list(history["driver_name"]) == ["A"]
