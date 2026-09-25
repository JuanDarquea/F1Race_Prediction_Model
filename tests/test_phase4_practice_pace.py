import pandas as pd
import pytest

import phase4_feature_engineering as phase4


def _write_session(root, session, drivers, lap_seconds):
    """Write a laps.csv keyed by abbreviation (as FastF1 does) plus results.csv."""
    folder = root / "2026" / "15_Azerbaijan_Grand_Prix" / session
    folder.mkdir(parents=True)
    rows = []
    for abbr, times in zip(drivers, lap_seconds):
        for lap, seconds in enumerate(times, start=1):
            rows.append(
                {
                    "Driver": abbr,
                    "DriverNumber": drivers[abbr]["number"],
                    "LapTimeSeconds": seconds,
                    "Season": 2026,
                    "RoundNumber": 15,
                    "EventName": "Azerbaijan Grand Prix",
                }
            )
    pd.DataFrame(rows).to_csv(folder / "laps.csv", index=False)
    pd.DataFrame(
        [
            {"Abbreviation": a, "DriverNumber": d["number"], "FullName": d["name"]}
            for a, d in drivers.items()
        ]
    ).to_csv(folder / "results.csv", index=False)


DRIVERS = {
    "NOR": {"number": 1, "name": "Lando Norris"},
    "ANT": {"number": 12, "name": "Kimi Antonelli"},
}


def test_practice_pace_uses_full_driver_names_so_it_can_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(phase4, "RAW_DIR", tmp_path)
    _write_session(tmp_path, "FP1", DRIVERS, [[104.0, 101.5, None], [103.0, 102.2]])
    _write_session(tmp_path, "FP2", DRIVERS, [[100.9, 101.1], [101.3, None]])

    pace = phase4._load_practice_pace().set_index("driver_name")["practice_pace"]

    assert set(pace.index) == {"Lando Norris", "Kimi Antonelli"}
    assert pace["Lando Norris"] == pytest.approx(100.9)  # best lap over FP1+FP2
    assert pace["Kimi Antonelli"] == pytest.approx(101.3)


def test_practice_pace_can_be_loaded_from_an_explicit_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(phase4, "RAW_DIR", tmp_path / "elsewhere")
    _write_session(tmp_path, "FP1", DRIVERS, [[101.0], [102.0]])
    pace = phase4._load_practice_pace(tmp_path / "2026")
    assert set(pace["driver_name"]) == {"Lando Norris", "Kimi Antonelli"}


def test_add_practice_rank_features_ranks_within_each_weekend():
    df = pd.DataFrame(
        {
            "season": [2026] * 4,
            "round": [1, 1, 2, 2],
            "practice_pace": [90.0, 91.0, 100.0, 99.0],
        }
    )
    out = phase4.add_practice_rank_features(df)
    assert out["practice_pace_rank"].tolist() == [1.0, 2.0, 2.0, 1.0]
    assert out["practice_pace_gap_to_best"].round(3).tolist() == [0.0, 1.0, 1.0, 0.0]
    assert out["practice_pace_percentile"].tolist() == [0.5, 1.0, 1.0, 0.5]
