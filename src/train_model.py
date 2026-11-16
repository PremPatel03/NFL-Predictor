import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Define which seasons to pull (2018-2024 for training, 2025 for testing)
SEASONS = list(range(2018, 2026))
FEATURE_COLS = [
    "pass_epa_diff",
    "run_epa_diff",
    "def_pass_epa_diff",
    "def_run_epa_diff",
    "rest_diff",
    "home_qb_out",
    "away_qb_out",
]

def load_and_engineer_features(seasons: list[int]) -> pd.DataFrame:
    print(f"Downloading play-by-play, schedules, and injuries for {seasons[0]}-{seasons[-1]}...")
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()
    injuries = nfl.load_injuries(seasons).to_pandas()

    # Keep only completed regular-season games
    games = schedules[
        (schedules["game_type"] == "REG") & (schedules["result"].notna())
    ].copy()
    games["home_win"] = (games["result"] > 0).astype(int)

    # --- A. Calculate Game-Level Offensive & Defensive EPA ---
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    off_epa = (
        plays.groupby(["season", "week", "posteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "off_pass_epa", "run": "off_run_epa"})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_epa = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    team_stats = pd.merge(off_epa, def_epa, on=["season", "week", "team"])
    team_stats = team_stats.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # --- B. Shift by 1 Game (Prevent Leakage) & Compute 5-Game EWMA ---
    epa_cols = ["off_pass_epa", "off_run_epa", "def_pass_epa", "def_run_epa"]
    for col in epa_cols:
        team_stats[f"ewma_{col}"] = (
            team_stats.groupby("team")[col]
            .transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
        )

    rolling_cols = ["season", "week", "team"] + [f"ewma_{c}" for c in epa_cols]
    rolling_df = team_stats[rolling_cols].dropna()

    # --- C. Historical QB Injury Flag ("Out" on Official Injury Report) ---
    qb_outs = (
        injuries[
            (injuries["position"] == "QB") & (injuries["report_status"] == "Out")
        ]
        .groupby(["season", "week", "team"])
        .size()
        .reset_index(name="qb_out_count")
    )
    qb_outs["qb_out"] = 1

    # --- D. Merge Home & Away Stats onto Game Schedule ---
    df = games.merge(
        rolling_df.add_prefix("home_"),
        left_on=["season", "week", "home_team"],
        right_on=["home_season", "home_week", "home_team"],
        how="inner",
    ).merge(
        rolling_df.add_prefix("away_"),
        left_on=["season", "week", "away_team"],
        right_on=["away_season", "away_week", "away_team"],
        how="inner",
    )

    # Merge QB injury flags (fill missing with 0 = healthy)
    df = df.merge(
        qb_outs[["season", "week", "team", "qb_out"]].rename(columns={"qb_out": "home_qb_out"}),
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"])

    df = df.merge(
        qb_outs[["season", "week", "team", "qb_out"]].rename(columns={"qb_out": "away_qb_out"}),
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"])

    df["home_qb_out"] = df["home_qb_out"].fillna(0)
    df["away_qb_out"] = df["away_qb_out"].fillna(0)

    # --- E. Matchup Differentials ---
    # Note: Negative defensive EPA is good for a defense, so positive diff favors Home
    df["pass_epa_diff"] = df["home_ewma_off_pass_epa"] - df["away_ewma_off_pass_epa"]
    df["run_epa_diff"] = df["home_ewma_off_run_epa"] - df["away_ewma_off_run_epa"]
    df["def_pass_epa_diff"] = df["away_ewma_def_pass_epa"] - df["home_ewma_def_pass_epa"]
    df["def_run_epa_diff"] = df["away_ewma_def_run_epa"] - df["home_ewma_def_run_epa"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # Skip Week 1 of each season where rolling stats carry over unevenly
    df = df[df["week"] >= 2].dropna(subset=FEATURE_COLS + ["home_win"])
    return df

def train_and_evaluate():
    df = load_and_engineer_features(SEASONS)

    # Chronological Train / Validation Split (Never shuffle sports data!)
    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[FEATURE_COLS], train_df["home_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["home_win"]

    # Build Pipeline: Standardize features -> L2-Regularized Logistic Regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    # Evaluate on unseen 2025 season
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n--- 2025 Out-of-Sample Validation Results ---")
    print(f"Games Evaluated : {len(test_df)}")
    print(f"Accuracy        : {accuracy_score(y_test, preds):.1%}")
    print(f"Log Loss        : {log_loss(y_test, probs):.4f} (lower is better; coin flip = 0.693)")
    print(f"Brier Score     : {brier_score_loss(y_test, probs):.4f} (lower is better; coin flip = 0.250)")

    # Inspect learned feature weights
    weights = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLS)
    print("\nLearned Feature Weights (Log-Odds):")
    print(weights.sort_values(ascending=False).round(3))

    # Save the trained artifact for our Sunday morning script
    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/nfl_win_prob_v1.joblib")
    print("\nSaved model artifact to models/nfl_win_prob_v1.joblib")

if __name__ == "__main__":
    train_and_evaluate()
