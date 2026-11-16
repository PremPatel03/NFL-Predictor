"""
Model training and chronological out-of-sample validation script for NFL game winners (v3).

Builds a leak-free historical dataset from 2018–2025 using:
  - Sequential pre-game Elo ratings with offseason mean reversion
  - 16-game rolling non-garbage-time EPA, success rate, and point differentials
  - Starting QB continuity flags (detecting week-to-week QB changes due to injury/benching)

Trains a standardized L2-regularized Logistic Regression model on 2018–2024, evaluates
out-of-sample performance on the full 2025 regular season, and saves the trained pipeline
to `models/nfl_win_prob_v3.joblib`.
"""

import joblib
import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    FEATURE_COLS,
    add_matchup_differentials,
    compute_elo_ratings,
    compute_team_rolling_stats,
)

# Training + validation seasons (2018-2024 = train, 2025 = out-of-sample holdout test)
SEASONS = list(range(2018, 2026))

def add_historical_qb_change_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Detects when a team starts a different QB than in its previous game of the same season.

    Why we track QB continuity instead of raw injury tables:
        In historical nflverse schedules, `home_qb_id` / `away_qb_id` records the QB who
        actually took the first snap. Comparing this week's starter against last week's
        starter reliably captures injuries, late scratches, and mid-season benchings.
    """
    reg = schedules[schedules["game_type"] == "REG"].sort_values(["season", "week"]).copy()

    # Reshape home and away starting QBs into a single team-by-team timeline
    home_qbs = reg[["season", "week", "home_team", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_qb_id": "qb_id"}
    )
    away_qbs = reg[["season", "week", "away_team", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_qb_id": "qb_id"}
    )
    all_qbs = pd.concat([home_qbs, away_qbs]).sort_values(["team", "season", "week"])

    # Look up the same team's starting QB from their prior game within the same season
    all_qbs["prev_qb_id"] = all_qbs.groupby(["team", "season"])["qb_id"].shift(1)

    # Flag 1 if a valid prior starter existed and changed this week; 0 otherwise
    all_qbs["qb_changed"] = (
        all_qbs["prev_qb_id"].notna() & (all_qbs["qb_id"] != all_qbs["prev_qb_id"])
    ).astype(int)

    # Merge home and away QB disruption flags back onto the main game schedule
    reg = reg.merge(
        all_qbs[["season", "week", "team", "qb_changed"]].rename(columns={"qb_changed": "home_qb_out"}),
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"])

    reg = reg.merge(
        all_qbs[["season", "week", "team", "qb_changed"]].rename(columns={"qb_changed": "away_qb_out"}),
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"])

    return reg

def train_and_evaluate():
    """
    Assembles the training/validation dataset, fits the Logistic Regression pipeline,
    prints 2025 out-of-sample accuracy/calibration metrics and learned feature weights,
    and serializes the artifact to disk.
    """
    print(f"Building v3 dataset (with Elo + QB starter continuity) for {SEASONS[0]}-{SEASONS[-1]}...")

    # Compute leak-free rolling team stats (shift_for_training=True shifts by 1 game)
    rolling_df = compute_team_rolling_stats(SEASONS, shift_for_training=True)
    schedules = nfl.load_schedules(SEASONS).to_pandas()

    # Compute sequential pre-game Elo ratings and starting QB disruption flags
    elo_df = compute_elo_ratings(schedules)
    games = add_historical_qb_change_flags(schedules)

    # Keep only completed games and define binary target (1 = home team won, 0 = otherwise)
    games = games[games["result"].notna()].copy()
    games["home_win"] = (games["result"] > 0).astype(int)

    # Merge pre-game Elo ratings, home team rolling stats, and away team rolling stats
    df = (
        games.merge(elo_df[["game_id", "elo_diff"]], on="game_id", how="inner")
        .merge(
            rolling_df.add_prefix("home_"),
            left_on=["season", "week", "home_team"],
            right_on=["home_season", "home_week", "home_team"],
            how="inner",
        )
        .merge(
            rolling_df.add_prefix("away_"),
            left_on=["season", "week", "away_team"],
            right_on=["away_season", "away_week", "away_team"],
            how="inner",
        )
    )

    # Compute home-minus-away differentials and exclude Week 1 (no in-season shifted history yet)
    df = add_matchup_differentials(df)
    df = df[df["week"] >= 2].dropna(subset=FEATURE_COLS + ["home_win"])

    # 4. Strict Chronological Train / Validation Split (never shuffle time-series sports data)
    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[FEATURE_COLS], train_df["home_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["home_win"]

    # Fit Standardized Logistic Regression Pipeline
    #    - StandardScaler puts Elo (~100s), EPA (~0.1s), and binary flags (0/1) on the same scale.
    #    - Strong L2 regularization (C=0.05) prevents overfitting across correlated team metrics.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.05, max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    # Evaluate Out-of-Sample Calibration & Classification Accuracy on 2025
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n--- 2025 Out-of-Sample Validation Results (v3) ---")
    print(f"Games Evaluated : {len(test_df)}")
    print(f"Accuracy        : {accuracy_score(y_test, preds):.1%} (was 59.8%)")
    print(f"Log Loss        : {log_loss(y_test, probs):.4f} (was 0.6416)")
    print(f"Brier Score     : {brier_score_loss(y_test, probs):.4f} (was 0.2264)")

    # Inspect standardized feature weights to confirm football domain alignment
    weights = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLS)
    print("\nLearned Feature Weights (Log-Odds):")
    print(weights.sort_values(ascending=False).round(3))

    # 7. Save model pipeline + exact feature list so inference uses identical column order
    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/nfl_win_prob_v3.joblib")
    print("\nSaved upgraded model to models/nfl_win_prob_v3.joblib")

if __name__ == "__main__":
    train_and_evaluate()
