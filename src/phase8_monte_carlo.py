from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from phase4_feature_engineering import build_feature_dataset, _find_lap_files
from phase5_model_training import TARGET_CONFIGS, _prepare_features
from phase6_predict_2026 import (
    _available_models,
    _load_model,
    _select_weekend,
    _prepare_training_columns,
    _align_to_columns,
    _prepare_features_inference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"
TRACK_TYPE_PATH = PROJECT_ROOT / "data" / "track_types.csv"
MC_OUTPUT_DIR = MODEL_DIR / "monte_carlo"

DEFAULT_SIMULATIONS = 10_000
RNG_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8: Monte Carlo Race Simulation")
    parser.add_argument(
        "--round", type=int, help="Round number (auto-detect latest if omitted)"
    )
    parser.add_argument(
        "--year", type=int, default=2026, help="Season year (default 2026)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help="Number of simulations (default 10000)",
    )
    return parser.parse_args()


def _derive_prediction_noise(
    full_features: pd.DataFrame, models: List[str]
) -> np.ndarray:
    """Compute residual distribution from model predictions on 2025 test data."""
    cfg = TARGET_CONFIGS["race"]

    test_data = full_features[full_features["season"] == 2025].copy()
    if test_data.empty:
        return np.array([0.0])

    try:
        feature_cols = _prepare_training_columns(full_features, "race")
    except ValueError:
        return np.array([0.0])

    prepared = _prepare_features(
        test_data, cfg.target_col, cfg.required_cols, cfg.drop_cols
    )
    if prepared.empty:
        return np.array([0.0])

    actual = prepared[cfg.target_col].values
    features = _align_to_columns(prepared, feature_cols)

    all_residuals: List[np.ndarray] = []
    for model_name in models:
        try:
            model = _load_model(model_name, cfg.output_suffix)
            predicted = model.predict(features)
            residuals = predicted - actual
            all_residuals.append(residuals)
        except Exception:
            continue

    if not all_residuals:
        return np.array([0.0])

    return np.concatenate(all_residuals)


def _derive_start_gains() -> Dict[int, np.ndarray]:
    """Compute historical lap 1 position gain/loss per grid position."""
    gains_by_grid: Dict[int, List[float]] = {}

    for lap_path in _find_lap_files("R"):
        try:
            laps = pd.read_csv(lap_path)
        except Exception:
            continue

        if "LapNumber" not in laps.columns or "Position" not in laps.columns:
            continue
        if "DriverNumber" not in laps.columns:
            continue

        lap1 = laps[laps["LapNumber"] == 1].copy()
        if lap1.empty:
            continue

        results_path = lap_path.parent / "results.csv"
        if not results_path.exists():
            continue

        try:
            results = pd.read_csv(results_path)
        except Exception:
            continue

        if (
            "GridPosition" not in results.columns
            or "DriverNumber" not in results.columns
        ):
            continue

        lap1 = lap1.copy()
        lap1["DriverNumber"] = lap1["DriverNumber"].astype(str)
        results = results.copy()
        results["DriverNumber"] = results["DriverNumber"].astype(str)

        merged = lap1.merge(
            results[["DriverNumber", "GridPosition"]],
            on="DriverNumber",
            how="inner",
        )

        for _, row in merged.iterrows():
            try:
                grid_pos = int(row["GridPosition"])
                lap1_pos = int(row["Position"])
                gain = lap1_pos - grid_pos
                gains_by_grid.setdefault(grid_pos, []).append(float(gain))
            except (ValueError, TypeError):
                continue

    return {gp: np.array(gains) for gp, gains in gains_by_grid.items()}


def _derive_dnf_rates(full_features: pd.DataFrame) -> Dict[str, float]:
    """Per-driver DNF rate from feature dataset."""
    if (
        "dnf_flag" not in full_features.columns
        or "driver_name" not in full_features.columns
    ):
        return {}

    rates = (
        full_features.groupby("driver_name", dropna=True)["dnf_flag"]
        .mean()
        .dropna()
        .to_dict()
    )
    return {str(k): float(v) for k, v in rates.items()}


def _derive_safety_car_prob() -> float:
    """Fraction of races with a safety car or virtual safety car."""
    sc_races = 0
    total_races = 0

    for lap_path in _find_lap_files("R"):
        try:
            laps = pd.read_csv(lap_path)
        except Exception:
            continue

        if "TrackStatus" not in laps.columns:
            continue

        total_races += 1
        statuses = laps["TrackStatus"].astype(str)
        has_sc = statuses.str.contains("4") | statuses.str.contains("6")
        if has_sc.any():
            sc_races += 1

    if total_races == 0:
        return 0.3

    return sc_races / total_races


def _derive_tire_parameters() -> Dict:
    """Derive compound distributions, stint lengths, and degradation rates."""
    compound_counts: Dict[str, int] = {}
    stint_lengths_raw: Dict[str, List[int]] = {}
    degradation_slopes: Dict[str, List[float]] = {}

    for lap_path in _find_lap_files("R"):
        try:
            laps = pd.read_csv(lap_path)
        except Exception:
            continue

        if "Compound" not in laps.columns or "Stint" not in laps.columns:
            continue

        # Identify driver column
        driver_col = None
        for col in ["DriverNumber", "Driver"]:
            if col in laps.columns:
                driver_col = col
                break
        if driver_col is None:
            continue

        if "LapTimeSeconds" not in laps.columns or "TyreLife" not in laps.columns:
            continue

        grouped = laps.groupby([driver_col, "Stint"], dropna=True)
        for _, stint_df in grouped:
            compound_series = stint_df["Compound"].dropna()
            if compound_series.empty:
                continue

            compound = str(compound_series.mode().iloc[0])
            length = len(stint_df)

            compound_counts[compound] = compound_counts.get(compound, 0) + 1
            stint_lengths_raw.setdefault(compound, []).append(length)

            valid = stint_df[["TyreLife", "LapTimeSeconds"]].dropna()
            valid = valid[
                (valid["LapTimeSeconds"] > 0) & np.isfinite(valid["LapTimeSeconds"])
            ]
            if len(valid) >= 3:
                try:
                    coeffs = np.polyfit(valid["TyreLife"], valid["LapTimeSeconds"], 1)
                    slope = float(coeffs[0])
                    if 0.0 <= slope <= 2.0:
                        degradation_slopes.setdefault(compound, []).append(slope)
                except (np.linalg.LinAlgError, ValueError):
                    pass

    total_stints = sum(compound_counts.values())
    if total_stints == 0:
        compound_weights: Dict[str, float] = {}
    else:
        compound_weights = {
            c: count / total_stints for c, count in compound_counts.items()
        }

    stint_lengths = {c: np.array(lengths) for c, lengths in stint_lengths_raw.items()}

    degradation_per_lap = {
        c: float(np.median(slopes))
        for c, slopes in degradation_slopes.items()
        if slopes
    }

    return {
        "compound_weights": compound_weights,
        "stint_lengths": stint_lengths,
        "degradation_per_lap": degradation_per_lap,
    }


def _derive_pit_time_distributions(
    full_features: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Per-team pit time distributions for sampling."""
    if (
        "avg_pit_time" not in full_features.columns
        or "team" not in full_features.columns
    ):
        return {}

    filtered = full_features[
        full_features["avg_pit_time"].notna() & (full_features["avg_pit_time"] > 0)
    ].copy()

    if filtered.empty:
        return {}

    result: Dict[str, np.ndarray] = {}
    for team, group in filtered.groupby("team", dropna=True):
        times = group["avg_pit_time"].dropna().values
        if len(times) > 0:
            result[str(team)] = np.array(times, dtype=float)

    return result


def _simulate_race(
    drivers: pd.DataFrame,
    predicted_positions: np.ndarray,
    noise_dist: np.ndarray,
    start_gains: Dict[int, np.ndarray],
    dnf_rates: Dict[str, float],
    safety_car_prob: float,
    tire_params: Dict,
    pit_distributions: Dict[str, np.ndarray],
    rng: np.random.Generator,
    weather_is_wet: bool = False,
) -> np.ndarray:
    """Run a single race simulation and return finishing positions (1-indexed, DNF = n+1)."""
    n_drivers = len(drivers)
    driver_names = drivers["driver_name"].tolist()
    teams = drivers["team"].tolist()

    # --- Phase 1: Qualifying ---
    # Sample noise from residual distribution and add to predicted positions
    if len(noise_dist) > 0:
        noise = rng.choice(noise_dist, size=n_drivers)
    else:
        noise = np.zeros(n_drivers)
    qualifying_scores = predicted_positions + noise
    # Rank to get grid positions (1-indexed, lower score = better)
    grid_positions = np.argsort(np.argsort(qualifying_scores)) + 1  # 1-indexed

    # --- Phase 2: Race Start ---
    post_start_positions = grid_positions.copy().astype(float)
    for i in range(n_drivers):
        grid_pos = int(grid_positions[i])
        if grid_pos in start_gains and len(start_gains[grid_pos]) > 0:
            gain = rng.choice(start_gains[grid_pos])
            post_start_positions[i] = grid_pos + gain
    # Re-rank after start
    post_start_positions = np.argsort(np.argsort(post_start_positions)) + 1  # 1-indexed

    # --- Phase 3: Stints (tire and pit time modelling) ---
    compound_weights: Dict[str, float] = tire_params.get("compound_weights", {})
    stint_lengths_dict: Dict[str, np.ndarray] = tire_params.get("stint_lengths", {})
    degradation_per_lap: Dict[str, float] = tire_params.get("degradation_per_lap", {})

    compounds = list(compound_weights.keys()) if compound_weights else ["MEDIUM"]
    weights_arr = (
        np.array([compound_weights[c] for c in compounds])
        if compound_weights
        else np.array([1.0])
    )
    weights_arr = weights_arr / weights_arr.sum()  # normalise

    cumulative_time = np.zeros(n_drivers)
    for i in range(n_drivers):
        team = teams[i]
        n_stints = int(rng.integers(2, 4))  # 2 or 3 stints
        driver_time = 0.0
        for stint_idx in range(n_stints):
            # Sample compound
            compound = str(rng.choice(compounds, p=weights_arr))
            # Sample stint length
            if compound in stint_lengths_dict and len(stint_lengths_dict[compound]) > 0:
                stint_len = int(rng.choice(stint_lengths_dict[compound]))
                stint_len = max(1, stint_len)
            else:
                stint_len = int(rng.integers(10, 25))
            # Compute tire degradation time contribution
            deg_rate = degradation_per_lap.get(compound, 0.05)
            tire_time = stint_len * deg_rate * (stint_len / 2.0)
            # Wet penalty for non-wet compounds
            if weather_is_wet and compound.upper() not in ("INTERMEDIATE", "WET"):
                tire_time *= 1.3
            driver_time += tire_time
            # Add pit stop time between stints (not after the last stint)
            if stint_idx < n_stints - 1:
                if team in pit_distributions and len(pit_distributions[team]) > 0:
                    pit_time = float(rng.choice(pit_distributions[team]))
                else:
                    pit_time = float(rng.integers(20, 30))  # fallback seconds
                driver_time += pit_time
        cumulative_time[i] = driver_time

    # --- Phase 4: Events (DNF + Safety Car) ---
    dnf_mask = np.zeros(n_drivers, dtype=bool)
    for i in range(n_drivers):
        name = driver_names[i]
        base_rate = dnf_rates.get(name, 0.05)
        effective_rate = base_rate * 1.5 if weather_is_wet else base_rate
        if rng.random() < effective_rate:
            dnf_mask[i] = True

    # Safety car compresses time gaps
    if rng.random() < safety_car_prob:
        cumulative_time = cumulative_time * 0.7

    # --- Phase 5: Final Ranking ---
    max_time = cumulative_time.max() if cumulative_time.max() > 0 else 1.0
    race_score = post_start_positions.astype(float) + (cumulative_time / max_time) * 5.0
    # DNF drivers get score 999
    race_score[dnf_mask] = 999.0

    # Rank non-DNF drivers first, then DNF drivers all get n_drivers + 1
    sorted_indices = np.argsort(race_score)
    finishing_positions = np.empty(n_drivers, dtype=int)
    rank = 1
    for idx in sorted_indices:
        if dnf_mask[idx]:
            finishing_positions[idx] = n_drivers + 1
        else:
            finishing_positions[idx] = rank
            rank += 1

    return finishing_positions


def run_monte_carlo(
    drivers: pd.DataFrame,
    predicted_positions: np.ndarray,
    noise_dist: np.ndarray,
    start_gains: Dict[int, np.ndarray],
    dnf_rates: Dict[str, float],
    safety_car_prob: float,
    tire_params: Dict,
    pit_distributions: Dict[str, np.ndarray],
    n_simulations: int = DEFAULT_SIMULATIONS,
    weather_is_wet: bool = False,
) -> pd.DataFrame:
    """Run N Monte Carlo race simulations and return a position-probability DataFrame."""
    rng = np.random.default_rng(RNG_SEED)
    n_drivers = len(drivers)
    driver_names = drivers["driver_name"].tolist()

    # counts[i, j] = number of times driver i finished in position j+1
    # Column index n_drivers represents DNF (position n+1)
    counts = np.zeros((n_drivers, n_drivers + 1), dtype=int)

    for _ in range(n_simulations):
        finishing_positions = _simulate_race(
            drivers=drivers,
            predicted_positions=predicted_positions,
            noise_dist=noise_dist,
            start_gains=start_gains,
            dnf_rates=dnf_rates,
            safety_car_prob=safety_car_prob,
            tire_params=tire_params,
            pit_distributions=pit_distributions,
            rng=rng,
            weather_is_wet=weather_is_wet,
        )
        for driver_idx, pos in enumerate(finishing_positions):
            # pos is 1-indexed; DNF = n_drivers + 1
            col = min(pos - 1, n_drivers)  # clamp to last column (DNF column)
            counts[driver_idx, col] += 1

    # Convert counts to probabilities
    probs = counts / n_simulations

    # Build result DataFrame
    position_cols = [f"P{p}" for p in range(1, n_drivers + 1)]
    records = []
    for i, name in enumerate(driver_names):
        row: Dict = {"driver_name": name}
        for p in range(1, n_drivers + 1):
            row[f"P{p}"] = probs[i, p - 1]
        row["DNF"] = probs[i, n_drivers]
        records.append(row)

    return pd.DataFrame(records, columns=["driver_name"] + position_cols + ["DNF"])


def _compute_summary(dist_df: pd.DataFrame, n_drivers: int) -> pd.DataFrame:
    """Compute summary statistics from the position distribution DataFrame."""
    records = []

    for _, row in dist_df.iterrows():
        name = row["driver_name"]

        # win_pct, podium_pct, points_pct
        win_pct = row.get("P1", 0.0) * 100.0
        podium_pct = sum(row.get(f"P{p}", 0.0) for p in range(1, 4)) * 100.0
        points_pct = sum(row.get(f"P{p}", 0.0) for p in range(1, 11)) * 100.0

        # expected_finish: weighted average over positions 1..n
        probs = np.array([row.get(f"P{p}", 0.0) for p in range(1, n_drivers + 1)])
        positions = np.arange(1, n_drivers + 1, dtype=float)
        prob_sum = probs.sum()
        if prob_sum > 0:
            expected_finish = float(np.dot(probs, positions) / prob_sum)
        else:
            expected_finish = float(n_drivers)

        # ci_90_low: position where cumulative probability reaches 5%
        # ci_90_high: position where cumulative probability reaches 95%
        cumprob = np.cumsum(probs)
        ci_low_indices = np.where(cumprob >= 0.05)[0]
        ci_high_indices = np.where(cumprob >= 0.95)[0]
        ci_90_low = int(ci_low_indices[0]) + 1 if len(ci_low_indices) > 0 else 1
        ci_90_high = (
            int(ci_high_indices[0]) + 1 if len(ci_high_indices) > 0 else n_drivers
        )

        dnf_pct = row.get("DNF", 0.0) * 100.0

        records.append(
            {
                "driver_name": name,
                "win_pct": win_pct,
                "podium_pct": podium_pct,
                "points_pct": points_pct,
                "expected_finish": expected_finish,
                "ci_90_low": ci_90_low,
                "ci_90_high": ci_90_high,
                "dnf_pct": dnf_pct,
            }
        )

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            "expected_finish", ascending=True
        ).reset_index(drop=True)
    return summary_df


def _save_outputs(
    dist_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    year: int,
    round_number: int,
    track: str,
    n_simulations: int,
    weather_is_wet: bool,
    warnings: List[str],
) -> None:
    """Save raw distribution CSV, summary CSV, and a text report."""
    MC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{year}_round{round_number:02d}"

    raw_path = MC_OUTPUT_DIR / f"simulation_raw_{tag}.csv"
    summary_path = MC_OUTPUT_DIR / f"simulation_summary_{tag}.csv"
    report_path = MC_OUTPUT_DIR / f"report_{tag}.txt"

    dist_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    weather_label = "Wet" if weather_is_wet else "Dry"
    top10 = summary_df.sort_values("expected_finish", ascending=True).head(10)

    lines: List[str] = []
    lines.append("Phase 8 Monte Carlo Simulation Report")
    lines.append("=" * 50)
    lines.append(f"Grand Prix : {track}")
    lines.append(f"Season     : {year}")
    lines.append(f"Round      : {round_number}")
    lines.append(f"Simulations: {n_simulations}")
    lines.append(f"Weather    : {weather_label}")
    lines.append("")

    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")

    # Table header
    header = f"{'Driver':<22} {'Win%':>6} {'Podium%':>8} {'Points%':>8} {'E[Pos]':>7} {'90% CI':>10} {'DNF%':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in top10.iterrows():
        ci_str = f"{int(row['ci_90_low'])}-{int(row['ci_90_high'])}"
        lines.append(
            f"{row['driver_name']:<22} {row['win_pct']:>6.1f} {row['podium_pct']:>8.1f} "
            f"{row['points_pct']:>8.1f} {row['expected_finish']:>7.2f} {ci_str:>10} {row['dnf_pct']:>6.1f}"
        )

    report_path.write_text("\n".join(lines) + "\n")

    print(f"  Raw distribution : {raw_path}")
    print(f"  Summary          : {summary_path}")
    print(f"  Report           : {report_path}")


def _plot_position_heatmap(
    dist_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    year: int,
    round_number: int,
    track: str,
) -> None:
    """Position distribution heatmap saved as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    MC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{year}_round{round_number:02d}"

    # Order drivers by expected finish
    ordered_drivers = summary_df.sort_values("expected_finish", ascending=True)[
        "driver_name"
    ].tolist()

    # Keep only position columns (P1..Pn), exclude DNF
    pos_cols = [c for c in dist_df.columns if c.startswith("P") and c != "DNF"]

    plot_df = dist_df.set_index("driver_name")[pos_cols].reindex(ordered_drivers)

    n_drivers = len(ordered_drivers)
    fig_height = max(8, n_drivers * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".0%",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )

    # Strip "P" prefix from x-axis labels
    ax.set_xticklabels([c.lstrip("P") for c in pos_cols], rotation=0)
    ax.set_title(
        f"Position Probability Distribution — {track} {year} (Round {round_number})"
    )
    ax.set_ylabel("Driver")
    ax.set_xlabel("Finishing Position")

    fig.tight_layout()
    out_path = MC_OUTPUT_DIR / f"position_heatmap_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Position heatmap : {out_path}")


def _plot_probability_bars(
    summary_df: pd.DataFrame,
    year: int,
    round_number: int,
    track: str,
) -> None:
    """Horizontal grouped bar chart for win/podium/points percentages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{year}_round{round_number:02d}"

    # Sort ascending so highest win% ends up at the top of horizontal bar chart
    plot_df = summary_df.sort_values("win_pct", ascending=True).reset_index(drop=True)
    drivers = plot_df["driver_name"].tolist()
    n_drivers = len(drivers)
    y = np.arange(n_drivers)

    bar_height = 0.25

    fig_height = max(8, n_drivers * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.barh(
        y - bar_height,
        plot_df["win_pct"],
        height=bar_height,
        color="#e74c3c",
        label="Win %",
    )
    ax.barh(
        y, plot_df["podium_pct"], height=bar_height, color="#f39c12", label="Podium %"
    )
    ax.barh(
        y + bar_height,
        plot_df["points_pct"],
        height=bar_height,
        color="#2ecc71",
        label="Points %",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(drivers)
    ax.set_xlabel("Probability (%)")
    ax.set_title(f"Race Outcome Probabilities — {track} {year} (Round {round_number})")
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = MC_OUTPUT_DIR / f"probability_bars_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Probability bars : {out_path}")


def main() -> None:
    # ------------------------------------------------------------------ #
    # Step 1: Parse args, print header                                     #
    # ------------------------------------------------------------------ #
    args = _parse_args()
    year: int = args.year
    n_simulations: int = args.simulations
    warnings: List[str] = []

    print(
        f"Phase 8 Monte Carlo — {year} Round {args.round}, {n_simulations} simulations"
    )

    # ------------------------------------------------------------------ #
    # Step 2: Load models                                                  #
    # ------------------------------------------------------------------ #
    models = _available_models()
    if not models:
        print("ERROR: No trained models found. Run Phase 5 first.")
        return

    # ------------------------------------------------------------------ #
    # Step 3: Build feature dataset                                        #
    # ------------------------------------------------------------------ #
    print("Building feature dataset…")
    full_features = build_feature_dataset(TRACK_TYPE_PATH)

    # ------------------------------------------------------------------ #
    # Step 4: Select weekend                                               #
    # ------------------------------------------------------------------ #
    round_number: Optional[int] = args.round

    # If no round supplied, try year; if year has no data, default to round 1
    if round_number is None:
        year_data = full_features[full_features["season"] == year]
        if year_data.empty:
            warnings.append(f"No data found for {year}; defaulting to round 1.")
            round_number = 1

    selected_round, track = _select_weekend(full_features, year, round_number)

    print(f"  Selected: {year} Round {selected_round} — {track}")

    # ------------------------------------------------------------------ #
    # Step 5: Get weekend driver list                                      #
    # ------------------------------------------------------------------ #
    weekend = full_features[
        (full_features["season"] == year) & (full_features["round"] == selected_round)
    ].copy()

    if weekend.empty:
        # Fallback to latest available season/round
        latest = (
            full_features[["season", "round"]]
            .drop_duplicates()
            .sort_values(["season", "round"])
            .iloc[-1]
        )
        fallback_season = int(latest["season"])
        fallback_round = int(latest["round"])
        warnings.append(
            f"No data for {year} Round {selected_round}; "
            f"falling back to {fallback_season} Round {fallback_round} driver list."
        )
        weekend = full_features[
            (full_features["season"] == fallback_season)
            & (full_features["round"] == fallback_round)
        ].copy()

    # ------------------------------------------------------------------ #
    # Step 6: Extract drivers DataFrame                                    #
    # ------------------------------------------------------------------ #
    driver_cols = ["driver_name", "team"]
    available_driver_cols = [c for c in driver_cols if c in weekend.columns]
    drivers = (
        weekend[available_driver_cols]
        .drop_duplicates(subset=["driver_name"])
        .reset_index(drop=True)
    )

    if "team" not in drivers.columns:
        drivers["team"] = "Unknown"

    print(f"  Drivers in weekend: {len(drivers)}")

    # ------------------------------------------------------------------ #
    # Step 7: Get model predictions                                        #
    # ------------------------------------------------------------------ #
    feature_cols = _prepare_training_columns(full_features, "race")
    inference_features = _prepare_features_inference(weekend, "race")
    aligned_features = _align_to_columns(inference_features, feature_cols)

    model = _load_model(models[0], TARGET_CONFIGS["race"].output_suffix)
    raw_predictions = model.predict(aligned_features)

    # Map predictions back to driver order via the weekend DataFrame index
    # (inference_features loses driver_name during one-hot encoding, but preserves index)
    driver_pred_map: Dict[str, float] = {}
    inference_driver_names = weekend.loc[
        inference_features.index, "driver_name"
    ].tolist()
    for name, pred in zip(inference_driver_names, raw_predictions.tolist()):
        driver_pred_map[str(name)] = float(pred)

    predicted_positions = np.array(
        [
            driver_pred_map.get(str(name), float(idx + 1))
            for idx, name in enumerate(drivers["driver_name"].tolist())
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------ #
    # Step 8: Derive simulation parameters                                 #
    # ------------------------------------------------------------------ #
    print("Deriving simulation parameters…")
    noise_dist = _derive_prediction_noise(full_features, models)
    start_gains = _derive_start_gains()
    dnf_rates = _derive_dnf_rates(full_features)
    safety_car_prob = _derive_safety_car_prob()
    tire_params = _derive_tire_parameters()
    pit_distributions = _derive_pit_time_distributions(full_features)

    # ------------------------------------------------------------------ #
    # Step 9: Detect wet weather                                           #
    # ------------------------------------------------------------------ #
    weather_is_wet = False
    if "race_is_wet" in weekend.columns:
        weather_is_wet = bool(weekend["race_is_wet"].any())

    # ------------------------------------------------------------------ #
    # Step 10: Print parameter summary                                     #
    # ------------------------------------------------------------------ #
    weather_label = "Wet" if weather_is_wet else "Dry"
    compounds = list(tire_params.get("compound_weights", {}).keys())
    compounds_str = ", ".join(compounds) if compounds else "N/A"
    print(f"  Safety car probability : {safety_car_prob:.1%}")
    print(f"  Weather                : {weather_label}")
    print(f"  Tire compounds         : {compounds_str}")

    # ------------------------------------------------------------------ #
    # Step 11: Run simulation                                              #
    # ------------------------------------------------------------------ #
    print(f"Running {n_simulations} Monte Carlo simulations…")
    dist_df = run_monte_carlo(
        drivers=drivers,
        predicted_positions=predicted_positions,
        noise_dist=noise_dist,
        start_gains=start_gains,
        dnf_rates=dnf_rates,
        safety_car_prob=safety_car_prob,
        tire_params=tire_params,
        pit_distributions=pit_distributions,
        n_simulations=n_simulations,
        weather_is_wet=weather_is_wet,
    )

    # ------------------------------------------------------------------ #
    # Step 12: Compute summary                                             #
    # ------------------------------------------------------------------ #
    summary_df = _compute_summary(dist_df, len(drivers))

    # ------------------------------------------------------------------ #
    # Step 13: Save outputs and plots                                      #
    # ------------------------------------------------------------------ #
    print("Saving outputs…")
    _save_outputs(
        dist_df,
        summary_df,
        year,
        selected_round,
        track,
        n_simulations,
        weather_is_wet,
        warnings,
    )
    _plot_position_heatmap(dist_df, summary_df, year, selected_round, track)
    _plot_probability_bars(summary_df, year, selected_round, track)

    # ------------------------------------------------------------------ #
    # Step 14: Print top-10 summary table to console                       #
    # ------------------------------------------------------------------ #
    top10 = summary_df.sort_values("expected_finish", ascending=True).head(10)
    header = f"\n{'Driver':<22} {'Win%':>6} {'Podium%':>8} {'Points%':>8} {'E[Pos]':>7} {'90% CI':>10} {'DNF%':>6}"
    print(header)
    print("-" * len(header.strip()))
    for _, row in top10.iterrows():
        ci_str = f"{int(row['ci_90_low'])}-{int(row['ci_90_high'])}"
        print(
            f"{row['driver_name']:<22} {row['win_pct']:>6.1f} {row['podium_pct']:>8.1f} "
            f"{row['points_pct']:>8.1f} {row['expected_finish']:>7.2f} {ci_str:>10} {row['dnf_pct']:>6.1f}"
        )

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
