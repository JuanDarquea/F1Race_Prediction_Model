"""Phase 9: Driver Elo / power rating.

Classic pairwise Elo: each race is decomposed into every pairwise driver
comparison (who finished ahead of whom), rated with a standard Elo update.
Keyed by driver_name (not driver_id — confirmed unstable across seasons in
this dataset) so a driver keeps their rating across a mid-season team change.
"""

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "feature_dataset.csv"
ELO_HISTORY_PATH = PROJECT_ROOT / "data" / "features" / "driver_elo_history.csv"
ELO_MODEL_DIR = PROJECT_ROOT / "models" / "elo"

DEFAULT_START_RATING = 1500.0
DEFAULT_K = 24.0
DEFAULT_SEASON_DECAY = 0.75


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that the driver with rating_a beats the driver with rating_b."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def pairwise_race_update(
    pre_ratings: Dict[str, float],
    finishing_positions: Dict[str, float],
    k: float = DEFAULT_K,
) -> Dict[str, float]:
    """Update ratings for one race using simultaneous pairwise comparisons.

    Every pair of drivers in finishing_positions is a head-to-head match
    decided by who has the lower (better) finishing position. Each driver's
    total delta across all their pairwise matches is averaged by their
    number of opponents, so K stays comparable regardless of grid size.
    """
    drivers = list(finishing_positions.keys())
    n = len(drivers)
    if n < 2:
        return dict(pre_ratings)

    deltas = {driver: 0.0 for driver in drivers}
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = drivers[i], drivers[j]
            r1, r2 = pre_ratings[d1], pre_ratings[d2]
            e1 = expected_score(r1, r2)
            e2 = 1.0 - e1
            p1, p2 = finishing_positions[d1], finishing_positions[d2]
            if p1 < p2:
                s1, s2 = 1.0, 0.0
            elif p1 > p2:
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5
            deltas[d1] += k * (s1 - e1)
            deltas[d2] += k * (s2 - e2)

    num_opponents = n - 1
    return {
        driver: pre_ratings[driver] + deltas[driver] / num_opponents
        for driver in drivers
    }


def apply_season_decay(
    ratings: Dict[str, float], decay: float = DEFAULT_SEASON_DECAY
) -> Dict[str, float]:
    """Regress every rating toward the field mean by (1 - decay) between seasons."""
    if not ratings:
        return {}
    mean_rating = sum(ratings.values()) / len(ratings)
    return {
        driver: mean_rating + decay * (rating - mean_rating)
        for driver, rating in ratings.items()
    }


def build_elo_history(
    df: pd.DataFrame,
    k: float = DEFAULT_K,
    decay: float = DEFAULT_SEASON_DECAY,
    start_rating: float = DEFAULT_START_RATING,
) -> pd.DataFrame:
    """Compute a walk-forward Elo history from a season/round/driver_name/race_position frame."""
    races = (
        df.dropna(subset=["race_position"])
        .loc[:, ["season", "round", "driver_name", "race_position"]]
        .copy()
    )
    races["season"] = races["season"].astype(int)
    races["round"] = races["round"].astype(int)

    ratings: Dict[str, float] = {}
    current_season = None
    rows = []

    race_keys = (
        races[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    )
    for _, key in race_keys.iterrows():
        season, round_ = int(key["season"]), int(key["round"])
        if current_season is not None and season != current_season:
            ratings = apply_season_decay(ratings, decay=decay)
        current_season = season

        race_df = races[(races["season"] == season) & (races["round"] == round_)]
        positions = dict(zip(race_df["driver_name"], race_df["race_position"]))
        for driver in positions:
            ratings.setdefault(driver, start_rating)
        pre_ratings = {driver: ratings[driver] for driver in positions}

        post_ratings = pairwise_race_update(pre_ratings, positions, k=k)
        for driver in positions:
            rows.append(
                {
                    "season": season,
                    "round": round_,
                    "driver_name": driver,
                    "elo_pre_race": pre_ratings[driver],
                    "elo_post_race": post_ratings[driver],
                }
            )
            ratings[driver] = post_ratings[driver]

    return pd.DataFrame(
        rows,
        columns=["season", "round", "driver_name", "elo_pre_race", "elo_post_race"],
    )


def _write_report(history: pd.DataFrame, report_path: Path) -> None:
    latest_season = int(history["season"].max())
    latest_round = int(history[history["season"] == latest_season]["round"].max())
    current = (
        history[
            (history["season"] == latest_season) & (history["round"] == latest_round)
        ]
        .sort_values("elo_post_race", ascending=False)
        .loc[:, ["driver_name", "elo_post_race"]]
    )

    season_start = (
        history[history["season"] == latest_season]
        .sort_values("round")
        .groupby("driver_name")["elo_pre_race"]
        .first()
    )
    movers = (
        (current.set_index("driver_name")["elo_post_race"] - season_start)
        .dropna()
        .sort_values(ascending=False)
    )

    with report_path.open("w") as f:
        f.write("Phase 9 - Driver Elo Report\n")
        f.write("============================\n\n")
        f.write(f"Ratings as of season {latest_season}, round {latest_round}\n\n")
        f.write("Current Ratings (highest first):\n")
        for _, row in current.iterrows():
            f.write(f"- {row['driver_name']}: {row['elo_post_race']:.1f}\n")
        f.write("\nBiggest gainers this season:\n")
        for driver, delta in movers.head(5).items():
            f.write(f"- {driver}: {delta:+.1f}\n")
        f.write("\nBiggest fallers this season:\n")
        for driver, delta in movers.tail(5).items():
            f.write(f"- {driver}: {delta:+.1f}\n")


def _write_trajectory_plots(history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = history["driver_name"].value_counts()
    eligible = counts[counts >= 10].index
    for driver in eligible:
        driver_history = history[history["driver_name"] == driver].sort_values(
            ["season", "round"]
        )
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(len(driver_history)), driver_history["elo_post_race"], marker="o"
        )
        plt.title(f"Elo trajectory - {driver}")
        plt.xlabel("Race index (chronological)")
        plt.ylabel("Elo rating (post-race)")
        plt.tight_layout()
        safe_name = driver.replace(" ", "_")
        plt.savefig(output_dir / f"trajectory_{safe_name}.png")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9: Driver Elo ratings")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--k-factor", type=float, default=DEFAULT_K)
    parser.add_argument("--season-decay", type=float, default=DEFAULT_SEASON_DECAY)
    args = parser.parse_args()

    ELO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ELO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    history = build_elo_history(df, k=args.k_factor, decay=args.season_decay)
    history.to_csv(ELO_HISTORY_PATH, index=False)
    _write_report(history, ELO_MODEL_DIR / "report.txt")
    _write_trajectory_plots(history, ELO_MODEL_DIR)
    print(f"Saved Elo history -> {ELO_HISTORY_PATH}")
    print(f"Saved report -> {ELO_MODEL_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
