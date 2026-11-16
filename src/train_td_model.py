"""
Anytime Touchdown probability training and chronological validation script.

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

SEASONS = list(range(2019, 2026))

def train_td_model() -> None:
    print(f"Building Anytime TD dataset (with Red-Zone & Goal-Line usage) for {SEASONS[0]}-{SEASONS[-1]}...")
    df = build_td_dataset(SEASONS, shift_for_training=True)

    # Filter to active rotation players (>= 2.5 rolling touches/targets per game, Week 2+)
    df = df[
        (df["week"] >= 2)
        & (df["ewma5_total_touches"] >= 2.5)
        & (df["total_touches"] > 0)
    ].dropna(subset=TD_FEATURES + ["scored_td"])

    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[TD_FEATURES], train_df["scored_td"]
    X_test, y_test = test_df[TD_FEATURES], test_df["scored_td"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- 2025 Out-of-Sample Anytime TD Validation ---")
    print(f"Player-Games Evaluated : {len(test_df)}")
    print(f"Base TD Rate           : {y_test.mean():.1%} (avg predicted: {probs.mean():.1%})")
    print(f"ROC AUC                : {roc_auc_score(y_test, probs):.3f}")
    print(f"Log Loss               : {log_loss(y_test, probs):.4f}")
    print(f"Brier Score            : {brier_score_loss(y_test, probs):.4f}")

    weights = pd.Series(model.named_steps["clf"].coef_[0], index=TD_FEATURES)
    print("\nLearned Feature Weights (Log-Odds):")
    print(weights.sort_values(ascending=False).round(3))

    joblib.dump(
        {"model": model, "features": TD_FEATURES},
        "models/anytime_td_v1.joblib",
    )
    print("\nSaved Anytime TD model to models/anytime_td_v1.joblib")

if __name__ == "__main__":
    train_td_model()
