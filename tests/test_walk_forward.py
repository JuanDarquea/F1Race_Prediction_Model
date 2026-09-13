import pandas as pd

from common.walk_forward import expanding_season_folds, latest_training_window


def test_expanding_season_folds_four_seasons():
    df = pd.DataFrame({"season": [2023, 2023, 2024, 2025, 2025, 2026]})
    folds = expanding_season_folds(df)
    assert folds == [
        ([2023], 2024),
        ([2023, 2024], 2025),
        ([2023, 2024, 2025], 2026),
    ]


def test_expanding_season_folds_single_season_has_no_folds():
    df = pd.DataFrame({"season": [2023, 2023, 2023]})
    assert expanding_season_folds(df) == []


def test_latest_training_window_returns_sorted_unique_seasons():
    df = pd.DataFrame({"season": [2024, 2023, 2023, 2025]})
    assert latest_training_window(df) == [2023, 2024, 2025]
