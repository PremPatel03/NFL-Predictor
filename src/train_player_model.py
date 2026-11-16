import joblib
import numpy as np
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

SEASONS = list(range(2019, 2026))

def _fit_and_save(df, feature_cols, target_col, artifact_path, label):
    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=50.0)),
    ])
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    upside_75_boost = float(np.percentile(y_test - preds, 75))

    print(f"\n--- 2025 Validation: {label} ---")
    print(f"Samples : {len(test_df)} | MAE: {mae:.2f} yds | R²: {r2:.3f} | 75th-Pct Boost: +{upside_75_boost:.1f} yds")

    joblib.dump(
        {"model": model, "features": feature_cols, "upside_75_boost": upside_75_boost},
        artifact_path,
    )
    print(f"Saved -> {artifact_path}")

def train_all_player_models():
    print(f"Building multi-position player datasets for {SEASONS[0]}-{SEASONS[-1]}...")
    datasets = build_all_player_datasets(SEASONS, shift_for_training=True)

    # WR/TE Receiving Yards
    wr_te = datasets["WR_TE"]
    wr_te = wr_te[(wr_te["week"] >= 2) & (wr_te["ewma5_targets"] >= 1.5) & (wr_te["targets"] > 0)].dropna(
        subset=WR_TE_FEATURES + ["rec_yards"]
    )
    _fit_and_save(wr_te, WR_TE_FEATURES, "rec_yards", "models/wr_te_yards_v1.joblib", "WR/TE Receiving Yards")

    # QB Passing Yards (Starting QBs with >= 12 rolling attempts/game)
    qbs = datasets["QB"]
    qbs = qbs[(qbs["week"] >= 2) & (qbs["ewma5_attempts"] >= 12.0) & (qbs["attempts"] >= 10)].dropna(
        subset=QB_FEATURES + ["pass_yards"]
    )
    _fit_and_save(qbs, QB_FEATURES, "pass_yards", "models/qb_pass_yards_v1.joblib", "QB Passing Yards")

    # 3. RB Rushing Yards (Active RBs with >= 4 rolling carries/game)
    rbs = datasets["RB"]
    rbs = rbs[(rbs["week"] >= 2) & (rbs["ewma5_carries"] >= 4.0) & (rbs["carries"] > 0)].dropna(
        subset=RB_FEATURES + ["rush_yards"]
    )
    _fit_and_save(rbs, RB_FEATURES, "rush_yards", "models/rb_rush_yards_v1.joblib", "RB Rushing Yards")

if __name__ == "__main__":
    train_all_player_models()
