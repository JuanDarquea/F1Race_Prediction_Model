"""Walk-forward season fold generation shared by Phase 9-11."""

from typing import List, Tuple

import pandas as pd


def expanding_season_folds(
    df: pd.DataFrame, season_col: str = "season"
) -> List[Tuple[List[int], int]]:
    """Return (train_seasons, test_season) pairs for an expanding walk-forward split.

    With seasons {2023, 2024, 2025, 2026} present in df, returns:
    [([2023], 2024), ([2023, 2024], 2025), ([2023, 2024, 2025], 2026)]
    The earliest season is never a test season (nothing to train on yet).
    """
    seasons = sorted(int(s) for s in df[season_col].dropna().unique())
    return [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]


def latest_training_window(df: pd.DataFrame, season_col: str = "season") -> List[int]:
    """Return every season present in df, for training a final 'live' model."""
    return sorted(int(s) for s in df[season_col].dropna().unique())
