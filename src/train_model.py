import joblib
import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, add_matchup_differentials, compute_team_rolling_stats

SEASONS = list(range(2018, 2026))

def train_and_evaluate():
    print(f"Building v2 dataset for {SEASONS[0]}-{SEASONS[-1]}...")
    rolling_df = compute_team_rolling_stats(SEASONS, shift_for_training=True)
    schedules = nfl.load_schedules(SEASONS).to_pandas()
    injuries = nfl.load_injuries(SEASONS).to_pandas()

    games = schedules[
        (schedules["game_type"] == "REG") & (schedules["result"].notna())
    ].copy()
    games["home_win"] = (games["result"] > 0).astype(int)

    # Starter-filtered historical QB injuries ("Out")
    qb_outs = injuries[
        (injuries["position"] == "QB") & (injuries["report_status"] == "Out")
    ][["season", "week", "team", "full_name"]].copy()
    qb_outs["out_name"] = qb_outs["full_name"].str.strip().str.lower()

    home_qb_out_set = set(zip(qb_outs["season"], qb_outs["week"], qb_outs["team"], qb_outs["out_name"]))

    games["home_qb_out"] = games.apply(
        lambda r: int(
            (r["season"], r["week"], r["home_team"], str(r.get("home_qb_name", "")).strip().lower())
            in home_qb_out_set
        ),
        axis=1,
    )
    games["away_qb_out"] = games.apply(
        lambda r: int(
            (r["season"], r["week"], r["away_team"], str(r.get("away_qb_name", "")).strip().lower())
            in home_qb_out_set
        ),
        axis=1,
    )

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

    df = add_matchup_differentials(df)
    df = df[df["week"] >= 2].dropna(subset=FEATURE_COLS + ["home_win"])

    # Chronological Train / Test Split
    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[FEATURE_COLS], train_df["home_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["home_win"]

    # Stronger L2 regularization (C=0.1) handles correlated 5-game and 16-game features cleanly
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n--- 2025 Out-of-Sample Validation Results (v2) ---")
    print(f"Games Evaluated : {len(test_df)}")
    print(f"Accuracy        : {accuracy_score(y_test, preds):.1%} (was 59.0%)")
    print(f"Log Loss        : {log_loss(y_test, probs):.4f} (was 0.6458)")
    print(f"Brier Score     : {brier_score_loss(y_test, probs):.4f} (was 0.2278)")

    weights = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLS)
    print("\nLearned Feature Weights (Log-Odds):")
    print(weights.sort_values(ascending=False).round(3))

    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/nfl_win_prob_v2.joblib")
    print("\nSaved upgraded model to models/nfl_win_prob_v2.joblib")

if __name__ == "__main__":
    train_and_evaluate()
