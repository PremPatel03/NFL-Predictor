"""
Anytime Touchdown probability training and chronological validation script.

Builds a leak-free historical dataset from 2019–2025 via `player_features.py` using:
  - 5-game rolling red-zone (<= 20 yds) and goal-line (<= 5 yds) opportunities
  - 5-game rolling total touches, scrimmage yards, and touchdowns scored
  - 8-game rolling opponent defensive TDs allowed per game
  - Vegas implied team totals, point spreads, and home-field flags

Trains a standardized, L2-regularized Logistic Regression classifier on 2019–2024
skill-position games (WR, TE, RB), evaluates out-of-sample probability calibration
(Brier Score, Log Loss, ROC AUC) on 2025, and saves `models/anytime_td_v1.joblib`.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from player_features import TD_FEATURES, build_td_dataset

# Training + validation window (2019-2024 = train, 2025 = out-of-sample holdout test)
SEASONS = list(range(2019, 2026))

def train_td_model() -> None:
    """
    Assembles the skill-position Anytime TD dataset, fits a calibrated Logistic Regression
    classifier on 2019–2024, prints 2025 out-of-sample discrimination and calibration
    metrics along with learned feature weights, and saves the trained artifact to disk.
    """
    print(f"Building Anytime TD dataset (with Red-Zone & Goal-Line usage) for {SEASONS[0]}-{SEASONS[-1]}...")

    # shift_for_training=True shifts rolling windows by 1 game so Week W only uses stats through Week W-1
    df = build_td_dataset(SEASONS, shift_for_training=True)

    # Filter to active rotation players (Week 2+, >= 2.5 rolling touches/targets per game,
    # and at least 1 touch/target in the evaluated game so inactive DNPs don't skew rates)
    df = df[
        (df["week"] >= 2)
        & (df["ewma5_total_touches"] >= 2.5)
        & (df["total_touches"] > 0)
    ].dropna(subset=TD_FEATURES + ["scored_td"])

    # Strict chronological train / validation split (never shuffle time-series sports data)
    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[TD_FEATURES], train_df["scored_td"]
    X_test, y_test = test_df[TD_FEATURES], test_df["scored_td"]

    # StandardScaler puts touches, yards, implied points, and binary flags on a unified scale;
    # L2 regularization (C=0.1) keeps predicted probabilities well-calibrated across correlated inputs.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    # Predict calibrated probability of scoring >= 1 rushing or receiving TD (class 1)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- 2025 Out-of-Sample Anytime TD Validation ---")
    print(f"Player-Games Evaluated : {len(test_df)}")
    print(f"Base TD Rate           : {y_test.mean():.1%} (avg predicted: {probs.mean():.1%})")
    print(f"ROC AUC                : {roc_auc_score(y_test, probs):.3f}")
    print(f"Log Loss               : {log_loss(y_test, probs):.4f}")
    print(f"Brier Score            : {brier_score_loss(y_test, probs):.4f}")

    # Inspect standardized feature weights to confirm domain alignment
    weights = pd.Series(model.named_steps["clf"].coef_[0], index=TD_FEATURES)
    print("\nLearned Feature Weights (Log-Odds):")
    print(weights.sort_values(ascending=False).round(3))

    # Serialize model pipeline and exact feature order for Sunday inference
    joblib.dump(
        {"model": model, "features": TD_FEATURES},
        "models/anytime_td_v1.joblib",
    )
    print("\nSaved Anytime TD model to models/anytime_td_v1.joblib")

if __name__ == "__main__":
    train_td_model()
