"""
Multi-position player yardage training and chronological validation script.

Builds leak-free historical player datasets from 2019–2025 via `player_features.py`,
trains separate standardized Ridge Regression models on 2019–2024 for:
  1. WR/TE Receiving Yards -> `models/wr_te_yards_v1.joblib`
  2. QB Passing Yards      -> `models/qb_pass_yards_v1.joblib`
  3. RB Rushing Yards      -> `models/rb_rush_yards_v1.joblib`

Evaluates each model out-of-sample on the 2025 season (reporting Mean Absolute Error,
R², and the 75th-percentile residual boost used for `ceiling_75` boom projections).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from player_features import (
    QB_FEATURES,
    RB_FEATURES,
    WR_TE_FEATURES,
    build_all_player_datasets,
)

# Training + validation window (2019-2024 = train, 2025 = out-of-sample holdout test)
SEASONS = list(range(2019, 2026))

def _fit_and_save(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    artifact_path: str,
    label: str,
) -> None:
    """
    Fits a standardized Ridge Regression model for a single position group, evaluates
    2025 out-of-sample accuracy, computes the 75th-percentile residual ceiling spread,
    and serializes the trained artifact to disk.

    Args:
        df: Filtered, leak-free feature DataFrame for the target position group.
        feature_cols: Exact ordered list of input feature names.
        target_col: Actual game yardage column (`rec_yards`, `pass_yards`, or `rush_yards`).
        artifact_path: Destination `.joblib` path inside `models/`.
        label: Human-readable position label for console logging.
    """
    # Strict chronological split: train on 2019-2024, validate out-of-sample on 2025
    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # StandardScaler normalizes features of different magnitudes (yards, shares, EPA);
    # Ridge(alpha=50.0) applies L2 regularization to prevent overfitting on correlated stats.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=50.0)),
    ])
    model.fit(X_train, y_train)

    # Clip negative predictions to 0 yards (a player cannot be projected for < 0 yards)
    preds = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Compute the 75th percentile of actual-minus-predicted residuals on the validation set.
    # Adding this spread to `proj_yards` estimates the yardage threshold a player clears ~25% of the time.
    upside_75_boost = float(np.percentile(y_test - preds, 75))

    print(f"\n--- 2025 Validation: {label} ---")
    print(f"Samples : {len(test_df)} | MAE: {mae:.2f} yds | R²: {r2:.3f} | 75th-Pct Boost: +{upside_75_boost:.1f} yds")

    # Save model pipeline, exact feature list, and calibrated 75th-percentile ceiling boost
    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "upside_75_boost": upside_75_boost,
        },
        artifact_path,
    )
    print(f"Saved -> {artifact_path}")

def train_all_player_models() -> None:
    """
    Builds historical player feature datasets, applies minimum usage filters so backup/bench
    noise doesn't skew the regression, and trains all three position yardage models.
    """
    print(f"Building multi-position player datasets for {SEASONS[0]}-{SEASONS[-1]}...")

    # shift_for_training=True shifts rolling windows by 1 game so Week W only uses stats through Week W-1
    datasets = build_all_player_datasets(SEASONS, shift_for_training=True)

    # WR/TE Receiving Yards: require Week 2+, >= 1.5 rolling targets/game, and at least 1 target in the game
    wr_te = datasets["WR_TE"]
    wr_te = wr_te[
        (wr_te["week"] >= 2)
        & (wr_te["ewma5_targets"] >= 1.5)
        & (wr_te["targets"] > 0)
    ].dropna(subset=WR_TE_FEATURES + ["rec_yards"])
    _fit_and_save(wr_te, WR_TE_FEATURES, "rec_yards", "models/wr_te_yards_v1.joblib", "WR/TE Receiving Yards")

    # QB Passing Yards: filter to starting-volume QBs (>= 12 rolling attempts/game, >= 10 in game)
    qbs = datasets["QB"]
    qbs = qbs[
        (qbs["week"] >= 2)
        & (qbs["ewma5_attempts"] >= 12.0)
        & (qbs["attempts"] >= 10)
    ].dropna(subset=QB_FEATURES + ["pass_yards"])
    _fit_and_save(qbs, QB_FEATURES, "pass_yards", "models/qb_pass_yards_v1.joblib", "QB Passing Yards")

    # RB Rushing Yards: filter to active backfield contributors (>= 4 rolling carries/game, >= 1 carry in game)
    rbs = datasets["RB"]
    rbs = rbs[
        (rbs["week"] >= 2)
        & (rbs["ewma5_carries"] >= 4.0)
        & (rbs["carries"] > 0)
    ].dropna(subset=RB_FEATURES + ["rush_yards"])
    _fit_and_save(rbs, RB_FEATURES, "rush_yards", "models/rb_rush_yards_v1.joblib", "RB Rushing Yards")

if __name__ == "__main__":
    train_all_player_models()
