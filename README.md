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

This project is a living document of my growth in AI and Data Science. As I refine the feature selection and explore more advanced ensemble methods, the model will be updated to reflect higher accuracy.

## Project Phases (Current Setup)

### Phase 1 - Data Collection
Script: `src/phase1_data_collection.py`

What it does:
- Installs and uses FastF1 cache
- Downloads sessions for multiple seasons
- Extracts laps, drivers, and positions
- Saves raw CSVs per session

Example:
```bash
python src/phase1_data_collection.py --years 2023 2024 2025 --sessions FP1 FP2 FP3 SQ S Q R
```

Outputs (example):
```
data/raw/fastf1/
  2022/01_Bahrain/FP1/laps.csv
  2022/01_Bahrain/Q/results.csv
  2022/01_Bahrain/R/drivers.csv
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

## Notes
- If Phase 2 fails, check that you already ran Phase 1 and that `data/raw/fastf1` exists.
- Driver names are preferred for analysis because driver numbers can change across seasons.
- Track types are stored in `data/track_types.csv` (street vs race circuits) and used in Phase 4.

## Troubleshooting
- Phase 1 errors about missing sessions: Some events do not have every session type. Try limiting sessions (e.g., `--sessions Q R`) or use fewer rounds with `--max-rounds`.
- Phase 2 “no race lap files found”: Run Phase 1 first and confirm `data/raw/fastf1/**/R/laps.csv` exists.
- Empty EDA outputs: Ensure you have both qualifying and race results in `data/raw/fastf1` (run Phase 1 with `--sessions Q R`).
