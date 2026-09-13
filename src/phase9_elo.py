"""Phase 9: Driver Elo / power rating.

Classic pairwise Elo: each race is decomposed into every pairwise driver
comparison (who finished ahead of whom), rated with a standard Elo update.
Keyed by driver_name (not driver_id — confirmed unstable across seasons in
this dataset) so a driver keeps their rating across a mid-season team change.
"""

from typing import Dict

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
