# F1 Race Result Prediction Model 🏎️💨
This repository contains a machine learning project designed to predict the outcomes of Formula 1 Grand Prix races. This project serves as a cornerstone of my learning path in Data Science and Software Engineering, focusing on data preprocessing, feature engineering, and predictive modeling.

## 1. Problem: The Complexity of the Paddock
Formula 1 is often described as "high-speed chess." Predicting a race winner is notoriously difficult because the outcome isn't just about the driver's skill. It is a multi-variant problem involving:

* Technical Variables: Car performance and reliability (Power Unit, Aerodynamics).
* Environmental Factors: Track temperature, weather conditions (rain vs. dry), and altitude.
* Strategic Decisions: Pit stop timing, tire compound selection (Soft, Medium, Hard), and fuel loads.
* Historical Context: Some drivers perform better on street circuits than on high-speed permanent tracks.
* Traditional statistics often fail to capture the "momentum" of a season, which is why a Machine Learning approach is necessary to weigh these variables effectively.

## 2. Data Source: FastF1 API
The backbone of this project is the FastF1 Python library, which provides an interface to access comprehensive Formula 1 data. Unlike standard databases, FastF1 allows for a deep dive into session-specific technical data.

Key data points utilized include:

- Lap Timing & Strategy: Detailed breakdown of every lap completed by every driver.
- Telemetry Data: Speed, throttle, and braking data (essential for understanding car performance).
- Tire Management: Tracking tire compounds, stint lengths, and degradation patterns.
- Weather & Track Conditions: Real-time atmospheric data during sessions.
- Session Results: Official classifications for Practice, Qualifying, and the Race.

## 3. Goal: Precision in Prediction
The primary objective of this project is to develop a model that can accurately predict the podium finishers (Top 3) or the outright winner of a given Grand Prix based on historical and live session data.

Learning Objectives:
1. Advanced Data Engineering: Handling complex API structures and converting telemetry/lap data into a "model-ready" format using Pandas and NumPy..
2. Feature Engineering: Creating unique metrics such as "Tire Advantage" (fresh vs. used tires) and "Driver Consistency" scores.
3. Model Evaluation: Testing various algorithms (e.g., Random Forest, XGBoost, or Gradient Boosting) to see which best handles the high variance of F1 results.
4. Software Best Practices: Implementing clean, modular code and utilizing Git for version control.

## Technical Stack
Language: Python

Data Source: FastF1 API

Libraries: Pandas, Scikit-Learn, Matplotlib, Seaborn

Version Control: Git & GitHub

```
"In Formula 1, everything is possible."
— Ayrton Senna
```

## 5. Results & Impact

### 5.1 Pipeline Architecture

The F1 prediction pipeline integrates data collection, feature engineering, multi-model training, and probabilistic simulation across 8 phases:

```
                    ╔═════════════════════════════════════════╗
                    ║     F1 PREDICTION PIPELINE (8 PHASES)   ║
                    ╚═════════════════════════════════════════╝

                              INPUT LAYER
                                  ↓
                    ┌─────────────────────────────┐
                    │   Phase 1: FastF1 API       │
                    │   (Data Collection)         │
                    └──────────────┬──────────────┘
                                   ↓
                    ╔══════════════════════════════╗
                    ║   PROCESSING LAYER           ║
                    ╚══════════════════════════════╝
                                   ↓
                    ┌─────────────────────────────┐
                    │   Phase 2: Data Cleaning    │
                    │   (Handle NaN, Outliers)    │
                    └──────────────┬──────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │   Phase 3: EDA              │
                    │   (Analysis & Visualization)│
                    └──────────────┬──────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │   Phase 4: Feature Eng.     │
                    │   (Create/Transform)        │
                    └──────────────┬──────────────┘
                                   ↓
                    ╔══════════════════════════════╗
                    ║   MODEL TRAINING LAYER       ║
                    ╚══════════════════════════════╝
                                   ↓
        ┌──────────────────────────┼──────────────────────────┐
        ↓                          ↓                          ↓
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │ Phase 5a:   │         │ Phase 5b:   │         │ Phase 5c:   │
   │ XGBoost     │         │ Random      │         │ Linear      │
   │ Model       │         │ Forest      │         │ Regression  │
   │             │         │ Model       │         │ Model       │
   └─────────────┘         └─────────────┘         └─────────────┘
        ↓                          ↓                          ↓
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │ Race        │         │ Race        │         │ Race        │
   │ Sprint      │         │ Sprint      │         │ Sprint      │
   │ Qualifying  │         │ Qualifying  │         │ Qualifying  │
   │ Predictions │         │ Predictions │         │ Predictions │
   └─────┬───────┘         └─────┬───────┘         └─────┬───────┘
         └──────────────────────┼──────────────────────┘
                                ↓
                    ╔══════════════════════════════╗
                    ║   OUTPUT LAYER               ║
                    ╚══════════════════════════════╝
                                   ↓
                    ┌─────────────────────────────┐
                    │ Phase 6: Ensemble Models    │
                    │ (Combine 3 Models)          │
                    │ → Voting/Weighting Strategy │
                    └──────────────┬──────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │ Phase 7: Model Evaluation   │
                    │ (Metrics & Validation)      │
                    └──────────────┬──────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │ Phase 8: Monte Carlo Sim.   │
                    │ (Race Simulation & Report)  │
                    └──────────────┬──────────────┘
                                   ↓
                    ╔═════════════════════════════╗
                    ║   FINAL PREDICTIONS & REPORT║
                    ╚═════════════════════════════╝
```

Each phase produces outputs used by downstream phases. Three models (XGBoost, Random Forest, Linear Regression) are trained in parallel on the same features, enabling ensemble-based predictions for robustness.

### 5.2 Performance Metrics (2025 Test Set)

| Model | Race MAE | Qualifying MAE | Top-10 Precision | Top-10 Recall |
|-------|----------|----------------|-----------------|---------------|
| XGBoost | 2.42 | 3.50 | 83% | 83% |
| RandomForest | 2.78 | 3.37 | 82% | 82% |
| LinearRegression | 2.86 | 3.54 | 82% | 82% |

XGBoost excels at capturing non-linear patterns in tire degradation and pit strategy timing, while ensemble approaches that combine multiple models improve robustness. Notice that qualifying predictions are significantly more accurate than race predictions—a single hot lap has less variance than a 58-lap race with dynamic strategy and incidents.

This project is a living document of my growth in AI and Data Science. As I refine the feature selection and explore more advanced ensemble methods, the model will be updated to reflect higher accuracy.

### 5.3 Case Study: Japan Grand Prix (Round 3, 2026)

The Japan Grand Prix demonstrates the model's strength on stable, predictable circuits. The model achieved exceptional accuracy with MAE of 1.4 positions and Spearman rank correlation of 0.867, with track conditions consistent throughout the race and conventional pit strategy execution.

| Driver | Pole % | Predicted Finish | Actual Finish | Error |
|--------|--------|-----------------|---------------|-------|
| Kimi Antonelli | 11.7% | 1 | 1 | 0 |
| Charles Leclerc | 1.1% | 2 | 3 | 1 |
| George Russell | 60.6% | 3 | 4 | 1 |
| Lando Norris | 0.2% | 4 | 5 | 1 |
| Oscar Piastri | 6.3% | 5 | 2 | 3 |
| Lewis Hamilton | 3.3% | 6 | 6 | 0 |
| Max Verstappen | 0.0% | 7 | 8 | 1 |
| Isack Hadjar | 16.2% | 8 | 12 | 4 |
| Pierre Gasly | 0.0% | 9 | 7 | 2 |
| Nico Hulkenberg | 0.0% | 10 | 11 | 1 |

**Performance Summary:**
- Mean Absolute Error (MAE): 1.4 positions
- Spearman Rank Correlation: 0.867
- Top-10 Precision: 100% (all 10 predicted drivers finished in top 20)
- Top-10 Recall: 80% (8 of 10 top finishers were in predictions; Lawson and Ocon were surprises)
- Key Result: 6 of 10 drivers had errors ≤ 1 position; top 3 showed strong ranking accuracy

This case study illustrates that ensemble predictions excel on stable, predictable circuits where historical patterns hold and track conditions remain consistent. Surprises like Oscar Piastri's strong 2nd place finish and lower-ranked drivers breaking into the top-10 indicate opportunities to refine feature engineering around team performance dynamics and driver form momentum.

### 5.4 Confidence & Caveats

While the ensemble achieves strong performance on 2025 data and the Japan Round 3 prediction, several factors limit generalization:

**Model Assumptions:**
- These predictions assume historical patterns hold. Unexpected weather shifts, crashes, or novel pit strategy can reshape outcomes.
- The model performs significantly better on stable circuits (Japan, Monaco) than chaotic ones (wet races, street circuits with high incident rates).
- Qualifying predictions are more accurate than race predictions due to lower variance in a single hot lap vs. a dynamic 58-lap race.

**Data Limitations:**
- Training data spans 2023–2024; 2026 regulations and car designs differ from historical seasons.
- Safety car frequency, weather severity, and unexpected failures are harder to predict than steady-state performance.

**Next Steps:**
- As 2026 races complete, evaluate model performance and retrain if needed.
- Consider specialized models for high-incident tracks (street circuits, wet-weather specialists).
- Expand feature set with real-time telemetry and pit stop data during race weekends.

## Quickstart
Run the full pipeline from raw data to Monte Carlo simulation:

```bash
# Note: Run Phase 1 year by year to avoid FastF1 API rate limits
python src/phase1_data_collection.py --years 2023 --sessions FP1 FP2 FP3 SQ S Q R
python src/phase1_data_collection.py --years 2024 --sessions FP1 FP2 FP3 SQ S Q R
python src/phase1_data_collection.py --years 2025 --sessions FP1 FP2 FP3 SQ S Q R
python src/phase1_data_collection.py --years 2026 --sessions FP1 FP2 FP3 SQ S Q R
python src/phase2_data_cleaning.py --sessions R S --split-by-year --aggregate-by-driver --aggregate-by-circuit
python src/phase3_eda.py
python src/phase4_feature_engineering.py --track-type-path data/track_types.csv
python src/phase5_model_training.py --top10-classification
python src/phase6_predict_2026.py --round 4
python src/phase7_evaluation.py
python src/phase8_monte_carlo.py --round 4 --year 2026 --simulations 10000
```

Interactive weekend prediction:
```bash
python src/phase5_predict_weekend.py
```

## Project Phases (Current Setup)

### Phase 1 - Data Collection
Script: `src/phase1_data_collection.py`

What it does:
- Installs and uses FastF1 cache
- Downloads sessions for multiple seasons
- Extracts laps, drivers, positions, and weather data
- Saves raw CSVs per session (laps.csv, results.csv, drivers.csv, weather.csv)

Example:
```bash
python src/phase1_data_collection.py --years 2023 2024 2025 --sessions FP1 FP2 FP3 SQ S Q R
```

Outputs (example):
```
data/raw/fastf1/
  2023/01_Bahrain/FP1/laps.csv
  2023/01_Bahrain/Q/results.csv
  2023/01_Bahrain/R/drivers.csv
  2023/01_Bahrain/R/weather.csv
```

### Phase 2 - Data Cleaning
Script: `src/phase2_data_cleaning.py`

What it does:
- Combines all race lap data into a single dataset
- Handles missing values (NaN lap times, DNFs)
- Standardizes key columns
- Adds `driver_name` for easier analysis
- Optional per-year and aggregated outputs

Example (basic clean):
```bash
python src/phase2_data_cleaning.py
```

Example (include sprint laps, split by year + aggregates):
```bash
python src/phase2_data_cleaning.py --sessions R S --split-by-year --aggregate-by-driver --aggregate-by-circuit
```

Outputs:
```
data/clean/fastf1_race_laps_clean.csv
data/clean/fastf1_race_laps_clean_2022.csv (optional)
data/clean/fastf1_race_laps_clean_by_driver.csv (optional)
data/clean/fastf1_race_laps_clean_by_circuit.csv (optional)
```

Standardized columns:
- driver
- driver_name
- team
- track
- grand_prix
- season
- round
- session
- position
- lap_time
- lap_number
- dnf

### Phase 3 - Exploratory Data Analysis (EDA)
Script: `src/phase3_eda.py`

What it does:
- Average finishing position per driver
- Team performance trends
- Track difficulty (DNF rate + lap time variation)
- Qualifying vs race correlation
- Driver consistency

Example:
```bash
python src/phase3_eda.py
```

Outputs:
```
data/eda/avg_finish_by_driver.csv
data/eda/avg_sprint_finish_by_driver.csv
data/eda/team_trends.csv
data/eda/track_difficulty.csv
data/eda/driver_consistency.csv
data/eda/avg_finish_positions.png
data/eda/avg_sprint_finish_positions.png
data/eda/lap_time_distribution.png
data/eda/qual_vs_race_scatter.png
data/eda/sprint_vs_race_scatter.png
data/eda/summary.txt
```

### Phase 4 - Feature Engineering
Script: `src/phase4_feature_engineering.py`

What it does:
- Builds per-driver/per-race feature rows
- Adds sprint weekend features (sprint result + sprint qualifying)
- Adds practice pace, track types, and rolling driver/team stats
- Extracts weather features (temperature, humidity, rainfall, wet flag)
- Extracts tire strategy features (stints, compound, tire life)
- Extracts pit stop features (avg pit time, total time lost)
- Applies tiered NaN imputation (group median > rolling historical > season median)

Example:
```bash
python src/phase4_feature_engineering.py
```

Outputs:
```
data/features/feature_dataset.csv
data/track_types.csv
```

Sprint weekend columns:
- sprint_position
- sprint_points
- sprint_qualifying_position

### Phase 5 - First ML Model
Script: `src/phase5_model_training.py`

What it does:
- Trains XGBoost, Random Forest, and Linear Regression on 2023–2024 and tests on 2025
- Produces models for race, sprint, and qualifying targets
- Predicts positions and ranks drivers per race
- Optional Top-10 classification output, confusion matrix, and visualization

Example (basic training):
```bash
python src/phase5_model_training.py
```

Example (Top-10 classification + confusion outputs):
```bash
python src/phase5_model_training.py --top10-classification
```

Interactive single-weekend predictions:
```bash
python src/phase5_predict_weekend.py
```

### Phase 6 - 2026 Race Week Prediction
Script: `src/phase6_predict_2026.py`

What it does:
- Refreshes raw data for a selected 2026 race weekend
- Rebuilds features and runs ensemble predictions (XGBoost + RF + Linear)
- Outputs Top-10 with Pole % and predicted finish position

Example:
```bash
python src/phase6_predict_2026.py --round 5
```

Output:
```
models/ensemble/predict_2026_round05.csv
models/ensemble/report_2026_round05.txt
```

Outputs:
```
models/xgboost/model.json
models/xgboost/predictions_2025.csv
models/xgboost/report.txt
models/xgboost/model_sprint.json
models/xgboost/predictions_2025_sprint.csv
models/xgboost/report_sprint.txt
models/xgboost/model_qualifying.json
models/xgboost/predictions_2025_qualifying.csv
models/xgboost/report_qualifying.txt
models/xgboost/top10_classification_2025.csv (optional)
models/xgboost/top10_confusion_by_race_2025.csv (optional)
models/xgboost/top10_confusion_matrix.png (optional)

models/random_forest/model.pkl
models/random_forest/model.json
models/random_forest/predictions_2025.csv
models/random_forest/report.txt
models/random_forest/model_sprint.pkl
models/random_forest/model_sprint.json
models/random_forest/predictions_2025_sprint.csv
models/random_forest/report_sprint.txt
models/random_forest/model_qualifying.pkl
models/random_forest/model_qualifying.json
models/random_forest/predictions_2025_qualifying.csv
models/random_forest/report_qualifying.txt
models/random_forest/top10_classification_2025.csv (optional)
models/random_forest/top10_confusion_by_race_2025.csv (optional)
models/random_forest/top10_confusion_matrix.png (optional)

models/linear_regression/model.pkl
models/linear_regression/model.json
models/linear_regression/predictions_2025.csv
models/linear_regression/report.txt
models/linear_regression/model_sprint.pkl
models/linear_regression/model_sprint.json
models/linear_regression/predictions_2025_sprint.csv
models/linear_regression/report_sprint.txt
models/linear_regression/model_qualifying.pkl
models/linear_regression/model_qualifying.json
models/linear_regression/predictions_2025_qualifying.csv
models/linear_regression/report_qualifying.txt
models/linear_regression/top10_classification_2025.csv (optional)
models/linear_regression/top10_confusion_by_race_2025.csv (optional)
models/linear_regression/top10_confusion_matrix.png (optional)

models/*/predict_2025_roundXX_race.csv (interactive)
models/*/predict_2025_roundXX_sprint.csv (interactive)
models/*/predict_2025_roundXX_qualifying.csv (interactive)
```

Prediction columns (2025):
- driver_name
- team
- season
- round
- grand_prix_name
- race_position
- predicted_position
- predicted_rank

### Phase 7 - Evaluation and Improvement
Script: `src/phase7_evaluation.py`

What it does:
- Evaluates Phase 5 prediction files (race/sprint/qualifying) for 2025
- Writes summary metrics (MAE, RMSE, Spearman, Top-10 precision/recall)
- Produces per-race breakdowns and driver error summaries
- Highlights largest errors for review

Example:
```bash
python src/phase7_evaluation.py
```

Outputs:
```
data/evaluation/summary_metrics_2025.csv
data/evaluation/per_race_metrics_2025.csv
data/evaluation/driver_error_summary_2025.csv
data/evaluation/largest_errors_2025.csv
data/evaluation/report_phase7.txt
data/evaluation/phase6_eval_2026.csv (if 2026 results are available)
data/evaluation/plots/mae_by_model_target_2025.png
```

### Phase 8 - Monte Carlo Race Simulation
Script: `src/phase8_monte_carlo.py`

What it does:
- Runs 10,000 (configurable) race simulations with randomness in qualifying, pit strategy, tire degradation, DNFs, and safety cars
- Derives simulation parameters from historical data (prediction noise, start gains, DNF rates, safety car frequency, tire compound distributions, pit time distributions)
- Models races in phases: Qualifying > Race Start > Tire Stints > Race Events > Final Ranking
- Outputs probability distributions over finishing positions for every driver

Example:
```bash
python src/phase8_monte_carlo.py --round 4 --year 2026 --simulations 10000
```

Outputs:
```
models/monte_carlo/simulation_raw_2026_round04.csv      (full probability matrix: 20 drivers x 20 positions)
models/monte_carlo/simulation_summary_2026_round04.csv   (win %, podium %, points %, expected finish, 90% CI, DNF %)
models/monte_carlo/report_2026_round04.txt               (human-readable report)
models/monte_carlo/position_heatmap_2026_round04.png     (position distribution heatmap)
models/monte_carlo/probability_bars_2026_round04.png     (win/podium/points bar chart)
```

Summary columns:
- win_pct, podium_pct, points_pct
- expected_finish (probability-weighted average position)
- ci_90_low, ci_90_high (90% confidence interval)
- dnf_pct (retirement probability)

## 8. Advanced Examples

### 8.1 Future Race Prediction: Miami Grand Prix (Round 4, 2026)

Miami's street circuit introduces higher unpredictability compared to Japan. Tight barriers mean one small error can end a driver's race, and safety cars are more frequent. The ensemble predictions and Monte Carlo simulations below reflect this uncertainty.

#### Ensemble Prediction (Phase 6)

| Driver | Pole % | Predicted Finish Position |
|--------|--------|---------------------------|
| Kimi Antonelli | 11.2% | 1 |
| Charles Leclerc | 0.9% | 2 |
| George Russell | 62.1% | 3 |
| Lando Norris | 0.2% | 4 |
| Oscar Piastri | 6.0% | 5 |
| Lewis Hamilton | 3.2% | 6 |
| Max Verstappen | 0.0% | 7 |
| Isack Hadjar | 15.9% | 8 |
| Pierre Gasly | 0.0% | 9 |
| Nico Hulkenberg | 0.0% | 10 |

#### Monte Carlo Simulation (Phase 8 - 10,000 iterations)

| Driver | Win % | Podium % | Expected Finish | 90% CI | DNF % |
|--------|-------|----------|-----------------|--------|-------|
| Kimi Antonelli | 23.5% | 50.4% | 3.2 | 1-22 | 22.5% |
| Charles Leclerc | 21.1% | 55.3% | 3.3 | 1-22 | 13.9% |
| George Russell | 19.0% | 50.3% | 3.7 | 1-22 | 10.6% |
| Oscar Piastri | 12.1% | 35.8% | 4.5 | 1-22 | 16.3% |
| Lando Norris | 8.4% | 32.2% | 4.9 | 1-22 | 12.3% |

**Interpretation:**
The Monte Carlo simulation (10,000 iterations) models qualifying, race start, tire stints, pit stops, DNFs, and safety cars. The wider confidence intervals and higher DNF percentages reflect Miami's chaotic nature compared to Japan's stable conditions. Notice how the Pole % and race predictions sometimes diverge—qualifying determines starting position, but strategy and incidents reshape the final order. The higher DNF percentages (10-23%) illustrate the increased risk of retirement on a street circuit.

## Notes
- If Phase 2 fails, check that you already ran Phase 1 and that `data/raw/fastf1` exists.
- Driver names are preferred for analysis because driver numbers can change across seasons.
- Track types are stored in `data/track_types.csv` (street vs race circuits) and used in Phase 4.

## Troubleshooting
- Phase 1 errors about missing sessions: Some events do not have every session type. Try limiting sessions (e.g., `--sessions Q R`) or use fewer rounds with `--max-rounds`.
- Phase 2 “no race lap files found”: Run Phase 1 first and confirm `data/raw/fastf1/**/R/laps.csv` exists.
- Empty EDA outputs: Ensure you have both qualifying and race results in `data/raw/fastf1` (run Phase 1 with `--sessions Q R`).
