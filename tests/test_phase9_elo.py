import pytest

from phase9_elo import (
    DEFAULT_K,
    apply_season_decay,
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
