import nflreadpy as nfl
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "pass_epa_diff_16",
    "run_epa_diff_16",
    "def_pass_epa_diff_16",
    "def_run_epa_diff_16",
    "pass_epa_diff_5",
    "success_rate_diff_16",
    "point_diff_16",
    "rest_diff",
    "is_home",
    "home_qb_out",
    "away_qb_out",
]

def compute_team_rolling_stats(seasons: list[int], shift_for_training: bool) -> pd.DataFrame:
    """Computes clean, non-garbage-time rolling team metrics across seasons."""
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    # 1. Filter to competitive plays (WP between 5% and 95%) to remove garbage-time noise
    plays = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["wp"].between(0.05, 0.95)
    ].copy()
    plays["success"] = (plays["epa"] > 0).astype(float)

    # Offensive EPA & Success Rate
    off = (
        plays.groupby(["season", "week", "posteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "off_pass_epa", "run": "off_run_epa"})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_sr = (
        plays.groupby(["season", "week", "posteam"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "success": "off_success_rate"})
    )

    # Defensive EPA
    defn = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Game-level Point Differential from completed schedules
    comp = schedules[schedules["result"].notna()].copy()
    home_scores = comp[["season", "week", "home_team", "result"]].rename(
        columns={"home_team": "team", "result": "point_diff"}
    )
    away_scores = comp[["season", "week", "away_team", "result"]].assign(
        point_diff=lambda d: -d["result"]
    )[["season", "week", "away_team", "point_diff"]].rename(columns={"away_team": "team"})
    margins = pd.concat([home_scores, away_scores], ignore_index=True)

    # Merge all per-team game stats
    stats = (
        off.merge(off_sr, on=["season", "week", "team"])
        .merge(defn, on=["season", "week", "team"])
        .merge(margins, on=["season", "week", "team"])
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

    # 2. Compute 16-game (long-term baseline) and 5-game (recent form) EWMA
    base_cols = [
        "off_pass_epa",
        "off_run_epa",
        "def_pass_epa",
        "def_run_epa",
        "off_success_rate",
        "point_diff",
    ]

    for col in base_cols:
        grouped = stats.groupby("team")[col]
        if shift_for_training:
            stats[f"ewma16_{col}"] = grouped.transform(
                lambda s: s.shift(1).ewm(span=16, min_periods=1).mean()
            )
            stats[f"ewma5_{col}"] = grouped.transform(
                lambda s: s.shift(1).ewm(span=5, min_periods=1).mean()
            )
        else:
            stats[f"ewma16_{col}"] = grouped.transform(
                lambda s: s.ewm(span=16, min_periods=1).mean()
            )
            stats[f"ewma5_{col}"] = grouped.transform(
                lambda s: s.ewm(span=5, min_periods=1).mean()
            )

    return stats

def add_matchup_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Computes home-minus-away differentials from merged home_ and away_ columns."""
    df["pass_epa_diff_16"] = df["home_ewma16_off_pass_epa"] - df["away_ewma16_off_pass_epa"]
    df["run_epa_diff_16"] = df["home_ewma16_off_run_epa"] - df["away_ewma16_off_run_epa"]
    df["def_pass_epa_diff_16"] = df["away_ewma16_def_pass_epa"] - df["home_ewma16_def_pass_epa"]
    df["def_run_epa_diff_16"] = df["away_ewma16_def_run_epa"] - df["home_ewma16_def_run_epa"]
    df["pass_epa_diff_5"] = df["home_ewma5_off_pass_epa"] - df["away_ewma5_off_pass_epa"]
    df["success_rate_diff_16"] = (
        df["home_ewma16_off_success_rate"] - df["away_ewma16_off_success_rate"]
    )
    df["point_diff_16"] = df["home_ewma16_point_diff"] - df["away_ewma16_point_diff"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["is_home"] = (df["location"] == "Home").astype(int)  # Handles international neutral-site games!
    return df
