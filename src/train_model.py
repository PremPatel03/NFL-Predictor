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

SEASONS = list(range(2018, 2026))

def add_historical_qb_change_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    """Detects when a team starts a different QB than its previous game (injury/benching)."""
    reg = schedules[schedules["game_type"] == "REG"].sort_values(["season", "week"]).copy()

    home_qbs = reg[["season", "week", "home_team", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_qb_id": "qb_id"}
    )
    away_qbs = reg[["season", "week", "away_team", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_qb_id": "qb_id"}
    )
    all_qbs = pd.concat([home_qbs, away_qbs]).sort_values(["team", "season", "week"])
    all_qbs["prev_qb_id"] = all_qbs.groupby(["team", "season"])["qb_id"].shift(1)
    all_qbs["qb_changed"] = (
        all_qbs["prev_qb_id"].notna() & (all_qbs["qb_id"] != all_qbs["prev_qb_id"])
    ).astype(int)

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
    print(f"Building v3 dataset (with Elo + QB starter continuity) for {SEASONS[0]}-{SEASONS[-1]}...")
    rolling_df = compute_team_rolling_stats(SEASONS, shift_for_training=True)
    schedules = nfl.load_schedules(SEASONS).to_pandas()

    elo_df = compute_elo_ratings(schedules)
    games = add_historical_qb_change_flags(schedules)
    games = games[games["result"].notna()].copy()
    games["home_win"] = (games["result"] > 0).astype(int)

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

    df = add_matchup_differentials(df)
    df = df[df["week"] >= 2].dropna(subset=FEATURE_COLS + ["home_win"])

    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[FEATURE_COLS], train_df["home_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["home_win"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.05, max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n--- 2025 Out-of-Sample Validation Results (v3) ---")
    print(f"Games Evaluated : {len(test_df)}")
    print(f"Accuracy        : {accuracy_score(y_test, preds):.1%} (was 59.8%)")
    print(f"Log Loss        : {log_loss(y_test, probs):.4f} (was 0.6416)")
    print(f"Brier Score     : {brier_score_loss(y_test, probs):.4f} (was 0.2264)")

    weights = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLS)
    print("\nLearned Feature Weights (Log-Odds):")
    print(weights.sort_values(ascending=False).round(3))

    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/nfl_win_prob_v3.joblib")
    print("\nSaved upgraded model to models/nfl_win_prob_v3.joblib")

if __name__ == "__main__":
    train_and_evaluate()
