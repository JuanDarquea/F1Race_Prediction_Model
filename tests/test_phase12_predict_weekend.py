import numpy as np
import pandas as pd
import pytest

from phase12_predict_weekend import (
    _build_weekend_features,
    _select_weekend,
    calibrate_qualifying,
    calibrate_winner_podium,
    fit_win_temperature,
    load_practice_pace,
    load_qualifying_grid,
    load_weekend_roster,
    load_track_type,
    normalize_podium_probabilities,
    predict_weekend,
    predicted_grid_order,
    summarize_oof,
    summarize_quali_oof,
    temper_win_probabilities,
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
        "p_pole",
        "p_quali_top3",
        "p_quali_top5",
    }
    assert expected_cols.issubset(predictions.columns)
    for col in (
        "p_win",
        "p_top3",
        "p_top10_race",
        "p_top10_qualifying",
        "p_pole",
        "p_quali_top3",
        "p_quali_top5",
    ):
        assert predictions[col].between(0.0, 1.0).all()


def test_temper_win_probabilities_sums_to_one_and_flattens():
    raw = np.array([0.9, 0.05, 0.01])
    sharp = temper_win_probabilities(raw, 1.0)
    flat = temper_win_probabilities(raw, 3.0)
    assert sharp.sum() == pytest.approx(1.0)
    assert flat.sum() == pytest.approx(1.0)
    assert flat.max() < sharp.max()


def test_fit_win_temperature_flattens_an_overconfident_model():
    rows = []
    for rnd in range(1, 21):
        winner_first = rnd % 2 == 0  # model always backs driver A, right half the time
        rows.append(("A", 1 if winner_first else 2, 0.95, rnd))
        rows.append(("B", 2 if winner_first else 1, 0.03, rnd))
    oof = pd.DataFrame(rows, columns=["driver_name", "race_position", "p_win", "round"])
    oof["season"] = 2025
    assert fit_win_temperature(oof) > 1.0


def test_fit_win_temperature_without_races_is_neutral():
    assert (
        fit_win_temperature(
            pd.DataFrame(columns=["season", "round", "p_win", "race_position"])
        )
        == 1.0
    )


def test_calibrate_winner_podium_maps_probabilities_and_keeps_raw():
    oof = pd.DataFrame(
        {
            "season": 2025,
            "round": [1] * 4 + [2] * 4,
            "driver_name": list("ABCD") * 2,
            "race_position": [1, 2, 3, 4] * 2,
            "p_win": [0.9, 0.05, 0.03, 0.02] * 2,
            "p_top3": [0.99, 0.9, 0.6, 0.1] * 2,
        }
    )
    live = pd.DataFrame(
        {
            "driver_name": list("ABCD"),
            "p_win": [0.9, 0.05, 0.03, 0.02],
            "p_top3": [0.99, 0.9, 0.6, 0.1],
        }
    )
    out, temperature = calibrate_winner_podium(live, oof)
    assert temperature >= 1.0
    assert out["p_win"].sum() == pytest.approx(1.0)
    assert {"p_win_raw", "p_top3_raw"}.issubset(out.columns)
    assert out["p_top3"].sum() == pytest.approx(3.0, abs=1e-6)
    assert out["p_top3"].between(0.0, 1.0).all()
    assert out["p_top3"].is_monotonic_decreasing


def test_calibrate_winner_podium_without_oof_keeps_raw_values():
    live = pd.DataFrame({"driver_name": ["A"], "p_win": [0.7], "p_top3": [0.8]})
    empty = pd.DataFrame(
        columns=["season", "round", "race_position", "p_win", "p_top3"]
    )
    out, temperature = calibrate_winner_podium(live, empty)
    assert temperature == 1.0
    assert out["p_win"].iloc[0] == pytest.approx(0.7)


def test_build_weekend_features_overrides_grid_with_real_qualifying():
    df = _synthetic_dataset()
    grid = {f"D{i}": float(12 - i) for i in range(11)}  # D11 missing from the grid
    weekend = _build_weekend_features(
        df, _synthetic_elo_history(), 2025, 3, "Baku", False, qualifying_grid=grid
    )
    by_driver = weekend.set_index("driver_name")["qualifying_position"]
    assert by_driver["D0"] == 12.0
    assert by_driver["D10"] == 2.0
    assert by_driver["D11"] == 13.0  # missing driver goes behind last classified


def test_load_qualifying_grid_reads_real_results_and_ignores_empty(tmp_path):
    round_dir = tmp_path / "2026" / "15_Azerbaijan_Grand_Prix" / "Q"
    round_dir.mkdir(parents=True)
    pd.DataFrame({"FullName": ["A B", "C D"], "Position": [1.0, 2.0]}).to_csv(
        round_dir / "results.csv", index=False
    )
    assert load_qualifying_grid(2026, 15, tmp_path) == {"A B": 1.0, "C D": 2.0}
    assert load_qualifying_grid(2026, 16, tmp_path) is None

    empty_dir = tmp_path / "2026" / "16_Bahrain_Grand_Prix" / "Q"
    empty_dir.mkdir(parents=True)
    pd.DataFrame({"FullName": ["A B"], "Position": [None]}).to_csv(
        empty_dir / "results.csv", index=False
    )
    assert load_qualifying_grid(2026, 16, tmp_path) is None


def test_normalize_podium_probabilities_sums_to_three_and_keeps_order():
    raw = np.array([0.99, 0.99, 0.99, 0.5, 0.2, 0.05, 0.01])
    out = normalize_podium_probabilities(raw)
    assert out.sum() == pytest.approx(3.0, abs=1e-6)
    assert (np.diff(out) <= 1e-12).all()
    assert out.max() < 1.0


def test_normalize_podium_probabilities_leaves_tiny_fields_untouched():
    raw = np.array([0.9, 0.8])
    assert normalize_podium_probabilities(raw).tolist() == raw.tolist()


def test_summarize_oof_counts_winner_hits_and_podium_overlap():
    oof = pd.DataFrame(
        {
            "season": 2025,
            "round": [1] * 4 + [2] * 4,
            "driver_name": list("ABCD") * 2,
            "race_position": [1, 2, 3, 4, 2, 1, 4, 3],
            "p_win": [0.7, 0.2, 0.05, 0.05, 0.7, 0.2, 0.05, 0.05],
            "p_top3": [0.9, 0.8, 0.7, 0.1, 0.9, 0.8, 0.1, 0.7],
        }
    )
    summary = summarize_oof(oof)
    assert summary["n_races"] == 2
    assert (summary["first_season"], summary["last_season"]) == (2025, 2025)
    assert summary["winner_top1_accuracy"] == pytest.approx(0.5)
    assert summary["podium_overlap"] == pytest.approx(1.0)


def test_build_weekend_features_rebuilds_practice_columns_from_real_laps():
    df = _synthetic_dataset()
    pace = {f"D{i}": 90.0 + (11 - i) for i in range(11)}  # D11 set no lap
    weekend = _build_weekend_features(
        df, _synthetic_elo_history(), 2025, 3, "Baku", False, practice_pace=pace
    )
    by_driver = weekend.set_index("driver_name")
    assert by_driver.loc["D10", "practice_pace_rank"] == 1.0  # fastest lap
    assert by_driver.loc["D0", "practice_pace_rank"] == 11.0
    assert by_driver.loc["D10", "practice_pace_gap_to_best"] == 0.0
    assert pd.isna(by_driver.loc["D11", "practice_pace"])  # missing stays missing


def test_load_practice_pace_reads_best_lap_across_sessions(tmp_path):
    for session, lap in (("FP1", 101.0), ("FP2", 100.5)):
        folder = tmp_path / "2026" / "15_Azerbaijan_Grand_Prix" / session
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "Driver": ["NOR"],
                "DriverNumber": [1],
                "LapTimeSeconds": [lap],
                "Season": [2026],
                "RoundNumber": [15],
                "EventName": ["Azerbaijan Grand Prix"],
            }
        ).to_csv(folder / "laps.csv", index=False)
        pd.DataFrame(
            {"Abbreviation": ["NOR"], "DriverNumber": [1], "FullName": ["Lando Norris"]}
        ).to_csv(folder / "results.csv", index=False)
    assert load_practice_pace(2026, 15, tmp_path) == {"Lando Norris": 100.5}
    assert load_practice_pace(2026, 16, tmp_path) is None


def test_build_weekend_features_recomputes_circuit_columns_for_the_target_track():
    df = _synthetic_dataset()
    df["track_type"] = "race"
    baku = df[(df["season"] == 2023) & (df["round"] == 1)].copy()
    baku["track"] = "Baku"
    baku["race_position"] = [
        float(12 - i) for i in range(12)
    ]  # D0 finished last at Baku
    df = pd.concat([df, baku], ignore_index=True)

    weekend = _build_weekend_features(
        df, _synthetic_elo_history(), 2025, 3, "Baku", False, track_type="street"
    )
    by_driver = weekend.set_index("driver_name")
    assert (weekend["track_type"] == "street").all()
    assert by_driver.loc["D0", "driver_performance_at_track"] == 12.0
    assert by_driver.loc["D11", "driver_performance_at_track"] == 1.0
    assert by_driver.loc["D0", "team_performance_at_track"] == pytest.approx(
        baku[baku["team"] == "Team0"]["race_position"].mean()
    )


def test_build_weekend_features_new_circuit_leaves_history_missing():
    df = _synthetic_dataset()
    weekend = _build_weekend_features(
        df, _synthetic_elo_history(), 2025, 3, "Brand New GP", False
    )
    assert weekend["driver_performance_at_track"].isna().all()


def test_load_track_type_reads_the_mapping(tmp_path):
    path = tmp_path / "track_types.csv"
    pd.DataFrame({"track": ["Azerbaijan Grand Prix"], "track_type": ["street"]}).to_csv(
        path, index=False
    )
    assert load_track_type("Azerbaijan Grand Prix", path) == "street"
    assert load_track_type("Unknown GP", path) is None
    assert load_track_type("Azerbaijan Grand Prix", tmp_path / "missing.csv") is None


def _quali_oof() -> pd.DataFrame:
    rows = []
    for rnd in (1, 2, 3):
        for i, name in enumerate("ABCDEFG"):
            rows.append(
                {
                    "season": 2025,
                    "round": rnd,
                    "driver_name": name,
                    "qualifying_position": (
                        i + 1 if rnd != 2 else (2, 1, 3, 4, 5, 6, 7)[i]
                    ),
                    "p_pole": [0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01][i],
                    "p_quali_top3": [0.95, 0.9, 0.6, 0.3, 0.1, 0.05, 0.02][i],
                    "p_quali_top5": [0.99, 0.97, 0.9, 0.7, 0.5, 0.1, 0.05][i],
                }
            )
    return pd.DataFrame(rows)


def test_calibrate_qualifying_is_nested_and_keeps_raw():
    oof = _quali_oof()
    live = oof[oof["round"] == 1][
        ["driver_name", "p_pole", "p_quali_top3", "p_quali_top5"]
    ].reset_index(drop=True)
    out, temperature = calibrate_qualifying(live, oof)
    assert temperature >= 1.0
    assert out["p_pole"].sum() == pytest.approx(1.0)
    assert {"p_pole_raw", "p_quali_top3_raw", "p_quali_top5_raw"}.issubset(out.columns)
    assert (out["p_quali_top3"] >= out["p_pole"] - 1e-12).all()
    assert (out["p_quali_top5"] >= out["p_quali_top3"] - 1e-12).all()


def test_calibrate_qualifying_without_oof_keeps_raw_values():
    live = pd.DataFrame(
        {
            "driver_name": ["A"],
            "p_pole": [0.4],
            "p_quali_top3": [0.7],
            "p_quali_top5": [0.9],
        }
    )
    empty = pd.DataFrame(
        columns=[
            "season",
            "round",
            "qualifying_position",
            "p_pole",
            "p_quali_top3",
            "p_quali_top5",
        ]
    )
    out, temperature = calibrate_qualifying(live, empty)
    assert temperature == 1.0
    assert out["p_pole"].iloc[0] == pytest.approx(0.4)


def test_summarize_quali_oof_scores_pole_and_overlaps():
    summary = summarize_quali_oof(_quali_oof())
    assert summary["n_sessions"] == 3
    assert summary["pole_accuracy"] == pytest.approx(2 / 3)  # round 2 pole was driver B
    assert summary["top3_overlap"] == pytest.approx(1.0)
    assert summary["top5_overlap"] == pytest.approx(1.0)
    assert (summary["first_season"], summary["last_season"]) == (2025, 2025)


def test_predicted_grid_order_ranks_each_slot_by_its_own_probability():
    frame = pd.DataFrame(
        {
            "driver_name": list("ABCDEFG"),
            # B has the best pole odds; A the best top-3 odds overall
            "p_pole": [0.20, 0.50, 0.10, 0.10, 0.05, 0.03, 0.02],
            "p_quali_top3": [0.90, 0.70, 0.60, 0.40, 0.20, 0.10, 0.10],
            "p_quali_top5": [0.95, 0.80, 0.75, 0.70, 0.65, 0.20, 0.15],
        }
    )
    order = predicted_grid_order(frame)
    assert order["driver_name"].tolist()[:5] == ["B", "A", "C", "D", "E"]
    assert order["grid_slot"].tolist() == list(range(1, 8))
    assert sorted(order.index) == sorted(frame.index)  # nobody lost or duplicated


def test_predict_weekend_with_predicted_grid_records_a_slot_for_every_driver():
    df = _synthetic_dataset()
    weekend = df[(df["season"] == 2025) & (df["round"] == 2)].copy()
    weekend["qualifying_position"] = 99.0  # stale value that must not be used
    predictions = predict_weekend(df, weekend, use_predicted_grid=True)
    assert sorted(predictions["predicted_grid_slot"]) == [
        float(i) for i in range(1, 13)
    ]
    assert predictions["p_win"].between(0.0, 1.0).all()


def test_summarize_quali_oof_reports_how_often_the_practice_leader_takes_pole():
    oof = _quali_oof()
    # practice leader is driver A every session; A took pole in rounds 1 and 3 only
    oof["practice_pace_rank"] = [1.0 if n == "A" else 2.0 for n in oof["driver_name"]]
    summary = summarize_quali_oof(oof)
    assert summary["practice_leader_pole_rate"] == pytest.approx(2 / 3)
    assert "practice_leader_pole_rate" not in summarize_quali_oof(_quali_oof())


def test_build_weekend_features_uses_the_real_roster_not_last_rounds_lineup():
    df = _synthetic_dataset()
    df["team_avg_finish"] = df["team"].map(
        {"Team0": 1.0, "Team1": 2.0, "Team2": 3.0, "Team3": 4.0}
    )
    # D3 sat out the latest round (still on the grid in round 1); D11 leaves.
    df = df[
        ~((df["season"] == 2025) & (df["round"] == 2) & (df["driver_name"] == "D3"))
    ]
    roster = {f"D{i}": f"Team{i % 4}" for i in range(11)}
    roster["D5"] = "Team0"  # D5 changed team
    weekend = _build_weekend_features(
        df, _synthetic_elo_history(), 2025, 3, "Baku", False, roster=roster
    )
    assert set(weekend["driver_name"]) == set(roster)
    by_driver = weekend.set_index("driver_name")
    assert by_driver.loc["D5", "team"] == "Team0"
    assert by_driver.loc["D5", "team_avg_finish"] == 1.0  # team stats follow the car
    assert (weekend["season"] == 2025).all() and (weekend["round"] == 3).all()
    assert weekend["driver_name"].is_unique


def test_load_weekend_roster_prefers_qualifying_then_latest_practice(tmp_path):
    base = tmp_path / "2026" / "15_Azerbaijan_Grand_Prix"
    for session, names in {"FP1": ["A B", "X Y"], "Q": ["A B", "C D"]}.items():
        (base / session).mkdir(parents=True)
        pd.DataFrame(
            {"FullName": names, "TeamName": ["Red Bull Racing", "Alpine"]}
        ).to_csv(base / session / "results.csv", index=False)
    assert load_weekend_roster(2026, 15, tmp_path) == {
        "A B": "Red Bull Racing",
        "C D": "Alpine",
    }
    (base / "Q" / "results.csv").unlink()
    assert set(load_weekend_roster(2026, 15, tmp_path)) == {"A B", "X Y"}
    assert load_weekend_roster(2026, 16, tmp_path) is None
