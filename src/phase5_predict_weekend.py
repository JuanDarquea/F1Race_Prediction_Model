import pandas as pd
from xgboost import XGBRegressor

from phase5_model_training import (
    FEATURE_PATH,
    IDENTIFIER_COLS,
    MODEL_DIR,
    TARGET_CONFIGS,
    _prepare_features,
    _split_data,
)


def _available_models() -> list[str]:
    candidates = ["xgboost", "random_forest", "linear_regression"]
    existing = []
    for name in candidates:
        if (MODEL_DIR / name).exists():
            existing.append(name)
    return existing


def _prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(prompt)
        for idx, opt in enumerate(options, start=1):
            print(f"{idx}. {opt}")
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.\n")
            continue
        choice_idx = int(choice)
        if 1 <= choice_idx <= len(options):
            return options[choice_idx - 1]
        print("Choice out of range.\n")


def _load_model(model_name: str, suffix: str):
    model_dir = MODEL_DIR / model_name
    if model_name == "xgboost":
        model_path = model_dir / f"model{suffix}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        model = XGBRegressor()
        model.load_model(model_path)
        return model

    model_path = model_dir / f"model{suffix}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    with model_path.open("rb") as f:
        return pd.read_pickle(f)


def _predict_target(
    model_name: str,
    target_key: str,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    selected_round: int,
    selected_track: str,
) -> pd.DataFrame:
    cfg = TARGET_CONFIGS[target_key]
    train = _prepare_features(
        train_raw, cfg.target_col, cfg.required_cols, cfg.drop_cols
    )
    test = _prepare_features(test_raw, cfg.target_col, cfg.required_cols, cfg.drop_cols)

    if train.empty or test.empty:
        raise ValueError(f"Not enough data to prepare {target_key} features.")

    train, test = train.align(test, join="left", axis=1, fill_value=0)

    feature_cols = [
        col
        for col in train.columns
        if col not in IDENTIFIER_COLS and col != cfg.target_col
    ]

    model = _load_model(model_name, cfg.output_suffix)

    round_idx = test_raw[
        (test_raw["season"] == 2025) & (test_raw["round"] == selected_round)
    ].index
    round_idx = round_idx.intersection(test.index)

    if round_idx.empty:
        raise ValueError("No rows found for the selected race weekend.")

    X_round = test.loc[round_idx, feature_cols]
    preds = model.predict(X_round)

    out = test_raw.loc[
        round_idx, ["driver_name", "team", "season", "round", "track"]
    ].copy()
    out = out.rename(columns={"track": "grand_prix_name"})
    if cfg.target_col in test_raw.columns:
        out[cfg.target_col] = test_raw.loc[round_idx, cfg.target_col]
    out["predicted_position"] = preds
    out["predicted_rank"] = out["predicted_position"].rank(method="first").astype(int)

    out = out.sort_values("predicted_rank")
    return out


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError("Feature dataset not found. Run Phase 4 first.")

    df = pd.read_csv(FEATURE_PATH)
    train_raw, test_raw = _split_data(df)

    if test_raw.empty:
        raise ValueError("No 2025 data found in feature dataset.")

    models = _available_models()
    if not models:
        raise ValueError("No trained models found. Run phase5_model_training.py first.")

    model_name = _prompt_choice("Select model:", models)

    races = (
        test_raw[["round", "track"]]
        .drop_duplicates()
        .sort_values("round")
        .reset_index(drop=True)
    )
    race_options = [
        f"Round {row.round} - {row.track}" for row in races.itertuples(index=False)
    ]
    race_choice = _prompt_choice("Select 2025 race weekend:", race_options)
    selected_idx = race_options.index(race_choice)
    selected_round = int(float(races.loc[selected_idx, "round"]))
    selected_track = races.loc[selected_idx, "track"]

    scope = _prompt_choice(
        "Select prediction scope:",
        [
            "Race top 10 only",
            "Race + Sprint (if available)",
            "Race + Sprint + Qualifying (if available)",
        ],
    )

    targets = ["race"]
    if scope in [
        "Race + Sprint (if available)",
        "Race + Sprint + Qualifying (if available)",
    ]:
        targets.append("sprint")
    if scope == "Race + Sprint + Qualifying (if available)":
        targets.append("qualifying")

    for target in targets:
        try:
            out = _predict_target(
                model_name, target, train_raw, test_raw, selected_round, selected_track
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{target}] {exc}")
            print("Run phase5_model_training.py to generate missing models.\n")
            continue

        top10 = out.head(10)
        output_path = (
            MODEL_DIR
            / model_name
            / f"predict_2025_round{selected_round:02d}_{target}.csv"
        )
        top10.to_csv(output_path, index=False)

        print(f"\n[{target.upper()}] {selected_track} (Round {selected_round})")
        print(top10[["driver_name", "team", "predicted_rank", "predicted_position"]])
        print(f"Saved -> {output_path}\n")


if __name__ == "__main__":
    main()
