"""
Multi-position feature engineering module for NFL player yardage projections.

Used by both `train_player_model.py` (historical training) and `predict_players.py`
(live Sunday inference) to build identical, leak-free feature tables for:
  1. WR/TE Receiving Yards
  2. QB Passing Yards
  3. RB Rushing Yards
"""

import nflreadpy as nfl
import numpy as np
import pandas as pd

# Feature Lists (Exact ordered columns expected by each position's Ridge model)
WR_TE_FEATURES = [
    "ewma5_rec_yards",                # 5-game rolling receiving yards
    "ewma5_targets",                  # 5-game rolling targets per game
    "ewma5_receptions",               # 5-game rolling catches per game
    "ewma5_target_share",             # 5-game rolling % of team pass attempts aimed at player
    "ewma5_air_yards_share",          # 5-game rolling % of team air yards (deep-threat usage)
    "ewma5_rec_epa",                  # 5-game rolling receiving Expected Points Added
    "opp_ewma8_wr_te_yards_allowed",  # Opponent defense's 8-game rolling WR/TE yards allowed
    "opp_ewma8_def_pass_epa",         # Opponent defense's 8-game rolling pass EPA allowed
    "implied_team_total",             # Vegas-implied points scored by the player's team
    "is_home",                        # 1 if playing in home stadium, 0 if road/neutral
    "is_wr",                          # 1 if Wide Receiver, 0 if Tight End
]

QB_FEATURES = [
    "ewma5_pass_yards",               # 5-game rolling passing yards
    "ewma5_attempts",                 # 5-game rolling pass attempts per game
    "ewma5_completions",              # 5-game rolling completions per game
    "ewma5_pass_epa",                 # 5-game rolling passing EPA
    "ewma5_yards_per_attempt",        # 5-game rolling passing efficiency (YPA)
    "opp_ewma8_pass_yards_allowed",   # Opponent defense's 8-game rolling QB passing yards allowed
    "opp_ewma8_def_pass_epa",         # Opponent defense's 8-game rolling pass EPA allowed
    "implied_team_total",             # Vegas-implied team points (higher = more passing upside)
    "team_spread",                    # Point spread (+ = favored, - = underdog needing to throw)
    "is_home",                        # 1 if home stadium, 0 if road/neutral
]

RB_FEATURES = [
    "ewma5_rush_yards",               # 5-game rolling rushing yards
    "ewma5_carries",                  # 5-game rolling rushing attempts per game
    "ewma5_rush_epa",                 # 5-game rolling rushing EPA
    "ewma5_yards_per_carry",          # 5-game rolling rushing efficiency (YPC)
    "ewma5_targets",                  # 5-game rolling receiving targets (proxy for 3-down snap share)
    "opp_ewma8_rush_yards_allowed",   # Opponent defense's 8-game rolling RB rushing yards allowed
    "opp_ewma8_def_run_epa",          # Opponent defense's 8-game rolling run EPA allowed
    "implied_team_total",             # Vegas-implied team points
    "team_spread",                    # Point spread (favored teams run more late in games)
    "is_home",                        # 1 if home stadium, 0 if road/neutral
]

def _add_rolling_ewma(
    df: pd.DataFrame,
    group_col: str,
    cols: list[str],
    span: int,
    shift: bool,
    prefix: str,
) -> pd.DataFrame:
    """
    Helper to compute Exponentially Weighted Moving Averages (EWMA) per player or defense.

    Args:
        df: Input DataFrame sorted chronologically by `group_col`, `season`, `week`.
        group_col: Entity ID to group by (e.g., `player_id` or `def_team`).
        cols: Raw stat columns to smooth.
        span: EWMA decay window (5 games for player usage, 8 games for opponent defense).
        shift: If True, shifts by 1 game so Week N only uses stats through Week N-1 (training).
               If False, uses all played games through last week (live Sunday predictions).
        prefix: Column name prefix for the generated rolling features.
    """
    for col in cols:
        grouped = df.groupby(group_col)[col]
        if shift:
            df[f"{prefix}{col}"] = grouped.transform(
                lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
            )
        else:
            df[f"{prefix}{col}"] = grouped.transform(
                lambda s: s.ewm(span=span, min_periods=1).mean()
            )
    return df

def _build_game_environment(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Converts game-level Vegas totals and point spreads into team-level game scripts.

    Formula:
        In nflverse schedules, `spread_line > 0` means the HOME team is favored.
        - Home Implied Points = (Total + Spread) / 2
        - Away Implied Points = (Total - Spread) / 2
    """
    reg = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "location"]
    ].copy()

    reg["home_implied_total"] = (reg["total_line"] + reg["spread_line"]) / 2.0
    reg["away_implied_total"] = (reg["total_line"] - reg["spread_line"]) / 2.0

    # Build home team environment rows
    home = reg[["season", "week", "home_team", "home_implied_total", "spread_line", "location"]].rename(
        columns={
            "home_team": "team",
            "home_implied_total": "implied_team_total",
            "spread_line": "team_spread",
        }
    )
    home["is_home"] = (home["location"] == "Home").astype(int)

    # Build away team environment rows (flip spread sign so + always means favored)
    away = reg[["season", "week", "away_team", "away_implied_total", "spread_line"]].rename(
        columns={"away_team": "team", "away_implied_total": "implied_team_total"}
    )
    away["team_spread"] = -away["spread_line"]
    away["is_home"] = 0

    return pd.concat(
        [home.drop(columns=["location"]), away.drop(columns=["spread_line"])],
        ignore_index=True,
    )

def build_all_player_datasets(seasons: list[int], shift_for_training: bool) -> dict[str, pd.DataFrame]:
    """
    Constructs clean, leak-free feature DataFrames for WR/TE, QB, and RB yardage models.

    Returns:
        Dictionary with keys `"WR_TE"`, `"QB"`, and `"RB"`, each mapping to a DataFrame
        containing smoothed player stats, opponent defensive generosity, and game script.
    """
    pstats = nfl.load_player_stats(seasons).to_pandas()
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    reg_stats = pstats[pstats["season_type"] == "REG"].copy()
    env_df = _build_game_environment(schedules)

    # Opponent Defensive Generosity Metrics (8-Game Rolling Window)
    plays = pbp[(pbp["season_type"] == "REG") & (pbp["play_type"].isin(["pass", "run"]))].copy()

    # Defensive EPA allowed per pass and per run play
    def_epa = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "def_team"})
    )

    # Total actual yards surrendered by each defense per game by position group
    wr_te_allowed = (
        reg_stats[reg_stats["position"].isin(["WR", "TE"])]
        .groupby(["season", "week", "opponent_team"])["receiving_yards"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "def_team", "receiving_yards": "wr_te_yards_allowed"})
    )
    pass_allowed = (
        reg_stats[reg_stats["position"] == "QB"]
        .groupby(["season", "week", "opponent_team"])["passing_yards"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "def_team", "passing_yards": "pass_yards_allowed"})
    )
    rush_allowed = (
        reg_stats[reg_stats["position"] == "RB"]
        .groupby(["season", "week", "opponent_team"])["rushing_yards"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "def_team", "rushing_yards": "rush_yards_allowed"})
    )

    def_df = (
        def_epa.merge(wr_te_allowed, on=["season", "week", "def_team"], how="left")
        .merge(pass_allowed, on=["season", "week", "def_team"], how="left")
        .merge(rush_allowed, on=["season", "week", "def_team"], how="left")
        .sort_values(["def_team", "season", "week"])
        .reset_index(drop=True)
    )
    def_cols = ["def_pass_epa", "def_run_epa", "wr_te_yards_allowed", "pass_yards_allowed", "rush_yards_allowed"]
    def_df = _add_rolling_ewma(def_df, "def_team", def_cols, span=8, shift=shift_for_training, prefix="opp_ewma8_")
    opp_keep = ["season", "week", "def_team"] + [f"opp_ewma8_{c}" for c in def_cols]

    # WR / TE Receiving Dataset

    wr_te = reg_stats[reg_stats["position"].isin(["WR", "TE"])].copy()
    for c in ["receiving_yards", "targets", "receptions", "target_share", "air_yards_share", "receiving_epa"]:
        wr_te[c] = wr_te[c].fillna(0.0)
    wr_te = wr_te.rename(columns={"receiving_yards": "rec_yards", "receiving_epa": "rec_epa"})
    wr_te = wr_te.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    wr_te = _add_rolling_ewma(
        wr_te,
        "player_id",
        ["rec_yards", "targets", "receptions", "target_share", "air_yards_share", "rec_epa"],
        span=5,
        shift=shift_for_training,
        prefix="ewma5_",
    )
    wr_te = (
        wr_te.merge(def_df[opp_keep], left_on=["season", "week", "opponent_team"], right_on=["season", "week", "def_team"], how="left")
        .merge(env_df, on=["season", "week", "team"], how="left")
    )
    # Fallback to league-average implied total (22.5) if betting lines aren't posted yet
    wr_te["implied_team_total"] = wr_te["implied_team_total"].fillna(22.5)
    wr_te["is_home"] = wr_te["is_home"].fillna(0)
    wr_te["is_wr"] = (wr_te["position"] == "WR").astype(int)

    # QB Passing Dataset

    qbs = reg_stats[reg_stats["position"] == "QB"].copy()
    for c in ["passing_yards", "attempts", "completions", "passing_epa"]:
        qbs[c] = qbs[c].fillna(0.0)
    qbs["yards_per_attempt"] = np.where(qbs["attempts"] > 0, qbs["passing_yards"] / qbs["attempts"], 0.0)
    qbs = qbs.rename(columns={"passing_yards": "pass_yards", "passing_epa": "pass_epa"})
    qbs = qbs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    qbs = _add_rolling_ewma(
        qbs,
        "player_id",
        ["pass_yards", "attempts", "completions", "pass_epa", "yards_per_attempt"],
        span=5,
        shift=shift_for_training,
        prefix="ewma5_",
    )
    qbs = (
        qbs.merge(def_df[opp_keep], left_on=["season", "week", "opponent_team"], right_on=["season", "week", "def_team"], how="left")
        .merge(env_df, on=["season", "week", "team"], how="left")
    )
    qbs["implied_team_total"] = qbs["implied_team_total"].fillna(22.5)
    qbs["team_spread"] = qbs["team_spread"].fillna(0.0)
    qbs["is_home"] = qbs["is_home"].fillna(0)

    # RB Rushing Dataset

    rbs = reg_stats[reg_stats["position"] == "RB"].copy()
    for c in ["rushing_yards", "carries", "rushing_epa", "targets"]:
        rbs[c] = rbs[c].fillna(0.0)
    rbs["yards_per_carry"] = np.where(rbs["carries"] > 0, rbs["rushing_yards"] / rbs["carries"], 0.0)
    rbs = rbs.rename(columns={"rushing_yards": "rush_yards", "rushing_epa": "rush_epa"})
    rbs = rbs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    rbs = _add_rolling_ewma(
        rbs,
        "player_id",
        ["rush_yards", "carries", "rush_epa", "yards_per_carry", "targets"],
        span=5,
        shift=shift_for_training,
        prefix="ewma5_",
    )
    rbs = (
        rbs.merge(def_df[opp_keep], left_on=["season", "week", "opponent_team"], right_on=["season", "week", "def_team"], how="left")
        .merge(env_df, on=["season", "week", "team"], how="left")
    )
    rbs["implied_team_total"] = rbs["implied_team_total"].fillna(22.5)
    rbs["team_spread"] = rbs["team_spread"].fillna(0.0)
    rbs["is_home"] = rbs["is_home"].fillna(0)

    return {"WR_TE": wr_te, "QB": qbs, "RB": rbs}

# Anytime Touchdown Feature List (WR, TE, RB)

TD_FEATURES = [
    "ewma5_rz_opps",            # 5-game rolling red-zone targets + carries (inside 20-yd line)
    "ewma5_gl_opps",            # 5-game rolling goal-line targets + carries (inside 5-yd line)
    "ewma5_total_tds",          # 5-game rolling rushing + receiving touchdowns
    "ewma5_total_touches",      # 5-game rolling total targets + carries (overall volume)
    "ewma5_total_yards",        # 5-game rolling scrimmage yards (rushing + receiving)
    "opp_ewma8_tds_allowed",    # Opponent defense's 8-game rolling skill-position TDs allowed
    "implied_team_total",       # Vegas-implied points scored by the player's team
    "team_spread",              # Point spread (+ = favored, - = underdog)
    "is_home",                  # 1 if playing at home, 0 if road/neutral
    "is_rb",                    # 1 if Running Back, 0 if WR/TE
]

def build_td_dataset(seasons: list[int], shift_for_training: bool) -> pd.DataFrame:
    """
    Builds a leak-free feature table for Anytime Touchdown prediction across WR, TE, and RB
    by combining weekly box scores with play-by-play red-zone (<=20 yds) and goal-line (<=5 yds) usage.
    """
    pstats = nfl.load_player_stats(seasons).to_pandas()
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    env_df = _build_game_environment(schedules)

    # Extract Red-Zone (<= 20 yds) & Goal-Line (<= 5 yds) Opportunities from Play-by-Play
    rz_plays = pbp[
        (pbp["season_type"] == "REG")
        & (pbp["play_type"].isin(["pass", "run"]))
        & (pbp["yardline_100"] <= 20)
    ].copy()

    rz_plays["is_gl"] = (rz_plays["yardline_100"] <= 5).astype(int)

    # Count red-zone & goal-line targets per receiver
    rz_rec = (
        rz_plays[rz_plays["play_type"] == "pass"]
        .dropna(subset=["receiver_player_id"])
        .groupby(["season", "week", "receiver_player_id"])
        .agg(rz_targets=("play_id", "count"), gl_targets=("is_gl", "sum"))
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )

    # Count red-zone & goal-line carries per rusher
    rz_rush = (
        rz_plays[rz_plays["play_type"] == "run"]
        .dropna(subset=["rusher_player_id"])
        .groupby(["season", "week", "rusher_player_id"])
        .agg(rz_carries=("play_id", "count"), gl_carries=("is_gl", "sum"))
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    # Filter Player Box Scores to Skill Positions (WR, TE, RB)
    skill = pstats[
        (pstats["season_type"] == "REG")
        & (pstats["position"].isin(["WR", "TE", "RB"]))
    ].copy()

    for col in ["receiving_tds", "rushing_tds", "targets", "carries", "receiving_yards", "rushing_yards"]:
        skill[col] = skill[col].fillna(0.0)

    skill["total_tds"] = skill["receiving_tds"] + skill["rushing_tds"]
    skill["scored_td"] = (skill["total_tds"] >= 1).astype(int)
    skill["total_touches"] = skill["targets"] + skill["carries"]
    skill["total_yards"] = skill["receiving_yards"] + skill["rushing_yards"]

    # Merge red-zone & goal-line usage onto player game rows
    skill = (
        skill.merge(rz_rec, on=["season", "week", "player_id"], how="left")
        .merge(rz_rush, on=["season", "week", "player_id"], how="left")
    )
    for col in ["rz_targets", "gl_targets", "rz_carries", "gl_carries"]:
        skill[col] = skill[col].fillna(0.0)

    skill["rz_opps"] = skill["rz_targets"] + skill["rz_carries"]
    skill["gl_opps"] = skill["gl_targets"] + skill["gl_carries"]

    # Opponent Defense: Rolling Skill-Position TDs Allowed per Game
    opp_tds = (
        skill.groupby(["season", "week", "opponent_team"])["total_tds"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "def_team", "total_tds": "tds_allowed"})
        .sort_values(["def_team", "season", "week"])
        .reset_index(drop=True)
    )
    opp_tds = _add_rolling_ewma(
        opp_tds, "def_team", ["tds_allowed"], span=8, shift=shift_for_training, prefix="opp_ewma8_"
    )

    # Smooth Player Usage & Red-Zone Opportunities (5-game EWMA)
    skill = skill.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    roll_cols = ["rz_opps", "gl_opps", "total_tds", "total_touches", "total_yards"]
    skill = _add_rolling_ewma(
        skill, "player_id", roll_cols, span=5, shift=shift_for_training, prefix="ewma5_"
    )

    # Merge opponent TD generosity and Vegas game environment
    skill = (
        skill.merge(
            opp_tds[["season", "week", "def_team", "opp_ewma8_tds_allowed"]],
            left_on=["season", "week", "opponent_team"],
            right_on=["season", "week", "def_team"],
            how="left",
        )
        .merge(env_df, on=["season", "week", "team"], how="left")
    )

    skill["implied_team_total"] = skill["implied_team_total"].fillna(22.5)
    skill["team_spread"] = skill["team_spread"].fillna(0.0)
    skill["is_home"] = skill["is_home"].fillna(0)
    skill["is_rb"] = (skill["position"] == "RB").astype(int)

    return skill
