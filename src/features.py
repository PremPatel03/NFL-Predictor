"""
Shared feature engineering module for the NFL game win-probability model (v3).

Both `train_model.py` (historical training) and `build_features.py` (Sunday live inference)
import from this file so feature definitions never drift out of sync.
"""

import nflreadpy as nfl
import numpy as np
import pandas as pd

# Exact ordered list of input features expected by the trained Logistic Regression pipeline.
# Every differential is oriented from the HOME team's perspective (positive = favors home team).
FEATURE_COLS = [
    "elo_diff",              # Pre-game Elo rating gap (includes home-field advantage)
    "pass_epa_diff_16",      # 16-game rolling offensive passing EPA differential
    "run_epa_diff_16",       # 16-game rolling offensive rushing EPA differential
    "def_pass_epa_diff_16",  # 16-game rolling defensive pass EPA differential (flipped so + = better home D)
    "def_run_epa_diff_16",   # 16-game rolling defensive run EPA differential (flipped so + = better home D)
    "success_rate_diff_16",  # 16-game rolling offensive play success rate (EPA > 0) differential
    "point_diff_16",         # 16-game rolling average point margin differential
    "rest_diff",             # Days of rest difference (home_rest - away_rest)
    "is_home",               # 1 if played at home team's stadium, 0 if neutral site (London/Munich/etc.)
    "home_qb_out",           # 1 if home starting QB changed / is ruled Out, else 0
    "away_qb_out",           # 1 if away starting QB changed / is ruled Out, else 0
]

def compute_elo_ratings(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sequential, leak-free pre-game Elo ratings across regular-season games.

    Why Elo helps:
    - Automatically adjusts for strength of schedule (beating a 1650 Elo team earns more
      credit than beating a 1350 Elo team).
    - Regresses teams 1/3 toward the 1500 league average between seasons to account for
      offseason roster turnover and draft/free-agency changes.
    """
    # Sort strictly chronologically so ratings update game-by-game in real order
    games = (
        schedules[schedules["game_type"] == "REG"]
        .sort_values(["season", "week", "gameday", "game_id"])
        .copy()
    )

    elos: dict[str, float] = {}  # Tracks each team's live Elo rating (starts at 1500.0)
    current_season = None
    records = []

    K = 20.0     # Step-size sensitivity: controls how fast ratings react to a single game
    HFA = 48.0   # Home-field advantage in Elo points (~2.0 real NFL points)

    for _, row in games.iterrows():
        season = row["season"]

        # At the start of each new NFL season, pull every team 1/3 back toward 1500
        if season != current_season:
            for team in elos:
                elos[team] = 1500.0 + (2.0 / 3.0) * (elos[team] - 1500.0)
            current_season = season

        home, away = row["home_team"], row["away_team"]
        home_elo = elos.get(home, 1500.0)
        away_elo = elos.get(away, 1500.0)

        # Only apply home-field advantage if the game is not at a neutral international venue
        hfa_adj = HFA if row["location"] == "Home" else 0.0
        elo_diff = (home_elo + hfa_adj) - away_elo

        # Record PRE-GAME ratings BEFORE looking at the game outcome (prevents data leakage)
        records.append({
            "game_id": row["game_id"],
            "home_pre_elo": home_elo,
            "away_pre_elo": away_elo,
            "elo_diff": elo_diff,
        })

        # Post-game update: only update Elo ratings if the game has actually been played
        if pd.notna(row["result"]):
            # Standard logistic win-expectancy curve (400 Elo points = ~10:1 odds)
            expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

            # Actual outcome: 1.0 for home win, 0.5 for tie, 0.0 for home loss
            actual_home = 1.0 if row["result"] > 0 else (0.5 if row["result"] == 0 else 0.0)

            # Margin-of-victory multiplier (FiveThirtyEight formula): rewards bigger wins
            # logarithmically while dampening blowouts by heavy favorites
            winner_elo_diff = elo_diff * (1 if actual_home == 1 else -1)
            mov_mult = np.log(abs(row["result"]) + 1.0) * (
                2.2 / (winner_elo_diff * 0.001 + 2.2)
            )

            # Zero-sum rating transfer from loser to winner
            shift = K * mov_mult * (actual_home - expected_home)
            elos[home] = home_elo + shift
            elos[away] = away_elo - shift

    return pd.DataFrame(records)

def compute_team_rolling_stats(seasons: list[int], shift_for_training: bool) -> pd.DataFrame:
    """
    Aggregates non-garbage-time play-by-play EPA, success rate, and point differential,
    then computes a 16-game Exponentially Weighted Moving Average (EWMA) per team.

    Args:
        seasons: List of NFL seasons to load (e.g., [2018, ..., 2025]).
        shift_for_training:
            - True when building historical training rows (shifts by 1 game so Week N
              only uses stats through Week N-1).
            - False on Sunday morning inference (uses all completed games through last week
              to project the upcoming unplayed slate).
    """
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    # Filter to competitive pass/run plays (Win Probability between 5% and 95%)
    #    This strips out 4th-quarter prevent defense, kneel-downs, and garbage-time stat padding.
    plays = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["wp"].between(0.05, 0.95)
    ].copy()

    # A "successful" play is any play that increases expected points (EPA > 0)
    plays["success"] = (plays["epa"] > 0).astype(float)

    # Offensive Efficiency: average EPA per pass and per run for each team-game
    off = (
        plays.groupby(["season", "week", "posteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "off_pass_epa", "run": "off_run_epa"})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Offensive Consistency: percentage of plays with positive EPA
    off_sr = (
        plays.groupby(["season", "week", "posteam"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "success": "off_success_rate"})
    )

    # Defensive Efficiency: average EPA allowed per pass and per run (lower = better defense)
    defn = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Scoreboard Margin: reshape schedule results into one row per team per game
    #    In nflverse schedules, `result = home_score - away_score`.
    comp = schedules[schedules["result"].notna()].copy()
    home_scores = comp[["season", "week", "home_team", "result"]].rename(
        columns={"home_team": "team", "result": "point_diff"}
    )
    away_scores = comp[["season", "week", "away_team", "result"]].assign(
        point_diff=lambda d: -d["result"]
    )[["season", "week", "away_team", "point_diff"]].rename(columns={"away_team": "team"})
    margins = pd.concat([home_scores, away_scores], ignore_index=True)

    # Combine all single-game team metrics and sort chronologically per team
    stats = (
        off.merge(off_sr, on=["season", "week", "team"])
        .merge(defn, on=["season", "week", "team"])
        .merge(margins, on=["season", "week", "team"])
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

    # Compute 16-game EWMA (roughly one full NFL season, weighting recent games higher)
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
            # Shift by 1 game so a row for Week W only sees games up to Week W-1
            stats[f"ewma16_{col}"] = grouped.transform(
                lambda s: s.shift(1).ewm(span=16, min_periods=1).mean()
            )
        else:
            # No shift needed when predicting future unplayed games
            stats[f"ewma16_{col}"] = grouped.transform(
                lambda s: s.ewm(span=16, min_periods=1).mean()
            )

    return stats

def add_matchup_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes head-to-head matchup differentials from merged `home_` and `away_` rolling columns.

    Convention: Positive values ALWAYS favor the home team.
    Note: For defensive EPA, lower is better, so we subtract `home - away` in reverse
    (`away_def_epa - home_def_epa`) so a positive number means the home defense is stingier.
    """
    df["pass_epa_diff_16"] = df["home_ewma16_off_pass_epa"] - df["away_ewma16_off_pass_epa"]
    df["run_epa_diff_16"] = df["home_ewma16_off_run_epa"] - df["away_ewma16_off_run_epa"]

    # Flipped sign: if away defense allows +0.10 EPA/play and home allows -0.05, diff = +0.15 (favors home)
    df["def_pass_epa_diff_16"] = df["away_ewma16_def_pass_epa"] - df["home_ewma16_def_pass_epa"]
    df["def_run_epa_diff_16"] = df["away_ewma16_def_run_epa"] - df["home_ewma16_def_run_epa"]

    df["success_rate_diff_16"] = (
        df["home_ewma16_off_success_rate"] - df["away_ewma16_off_success_rate"]
    )
    df["point_diff_16"] = df["home_ewma16_point_diff"] - df["away_ewma16_point_diff"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # Neutral-site international games have location == "Neutral" -> is_home = 0
    df["is_home"] = (df["location"] == "Home").astype(int)
    return df
