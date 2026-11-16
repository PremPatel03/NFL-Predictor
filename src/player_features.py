import nflreadpy as nfl
import numpy as np
import pandas as pd

WR_TE_FEATURES = [
    "ewma5_rec_yards",
    "ewma5_targets",
    "ewma5_receptions",
    "ewma5_target_share",
    "ewma5_air_yards_share",
    "ewma5_rec_epa",
    "opp_ewma8_wr_te_yards_allowed",
    "opp_ewma8_def_pass_epa",
    "implied_team_total",
    "is_home",
    "is_wr",
]

QB_FEATURES = [
    "ewma5_pass_yards",
    "ewma5_attempts",
    "ewma5_completions",
    "ewma5_pass_epa",
    "ewma5_yards_per_attempt",
    "opp_ewma8_pass_yards_allowed",
    "opp_ewma8_def_pass_epa",
    "implied_team_total",
    "team_spread",
    "is_home",
]

RB_FEATURES = [
    "ewma5_rush_yards",
    "ewma5_carries",
    "ewma5_rush_epa",
    "ewma5_yards_per_carry",
    "ewma5_targets",
    "opp_ewma8_rush_yards_allowed",
    "opp_ewma8_def_run_epa",
    "implied_team_total",
    "team_spread",
    "is_home",
]

def _add_rolling_ewma(df: pd.DataFrame, group_col: str, cols: list[str], span: int, shift: bool, prefix: str) -> pd.DataFrame:
    for col in cols:
        grouped = df.groupby(group_col)[col]
        if shift:
            df[f"{prefix}{col}"] = grouped.transform(lambda s: s.shift(1).ewm(span=span, min_periods=1).mean())
        else:
            df[f"{prefix}{col}"] = grouped.transform(lambda s: s.ewm(span=span, min_periods=1).mean())
    return df

def _build_game_environment(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "location"]
    ].copy()

    # Positive spread_line = home team favored
    reg["home_implied_total"] = (reg["total_line"] + reg["spread_line"]) / 2.0
    reg["away_implied_total"] = (reg["total_line"] - reg["spread_line"]) / 2.0

    home = reg[["season", "week", "home_team", "home_implied_total", "spread_line", "location"]].rename(
        columns={"home_team": "team", "home_implied_total": "implied_team_total", "spread_line": "team_spread"}
    )
    home["is_home"] = (home["location"] == "Home").astype(int)

    away = reg[["season", "week", "away_team", "away_implied_total", "spread_line"]].rename(
        columns={"away_team": "team", "away_implied_total": "implied_team_total"}
    )
    away["team_spread"] = -away["spread_line"]
    away["is_home"] = 0

    return pd.concat([home.drop(columns=["location"]), away.drop(columns=["spread_line"])], ignore_index=True)

def build_all_player_datasets(seasons: list[int], shift_for_training: bool) -> dict[str, pd.DataFrame]:
    """Builds leak-free feature tables for WR/TE receiving, QB passing, and RB rushing."""
    pstats = nfl.load_player_stats(seasons).to_pandas()
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    reg_stats = pstats[pstats["season_type"] == "REG"].copy()
    env_df = _build_game_environment(schedules)

    # Opponent Defensive Metrics from PBP & Box Scores
    plays = pbp[(pbp["season_type"] == "REG") & (pbp["play_type"].isin(["pass", "run"]))].copy()
    def_epa = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "def_team"})
    )

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

    # WR/TE Receiving Dataset
    wr_te = reg_stats[reg_stats["position"].isin(["WR", "TE"])].copy()
    for c in ["receiving_yards", "targets", "receptions", "target_share", "air_yards_share", "receiving_epa"]:
        wr_te[c] = wr_te[c].fillna(0.0)
    wr_te = wr_te.rename(columns={"receiving_yards": "rec_yards", "receiving_epa": "rec_epa"})
    wr_te = wr_te.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    wr_te = _add_rolling_ewma(
        wr_te, "player_id",
        ["rec_yards", "targets", "receptions", "target_share", "air_yards_share", "rec_epa"],
        span=5, shift=shift_for_training, prefix="ewma5_"
    )
    wr_te = (
        wr_te.merge(def_df[opp_keep], left_on=["season", "week", "opponent_team"], right_on=["season", "week", "def_team"], how="left")
        .merge(env_df, on=["season", "week", "team"], how="left")
    )
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
        qbs, "player_id",
        ["pass_yards", "attempts", "completions", "pass_epa", "yards_per_attempt"],
        span=5, shift=shift_for_training, prefix="ewma5_"
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
        rbs, "player_id",
        ["rush_yards", "carries", "rush_epa", "yards_per_carry", "targets"],
        span=5, shift=shift_for_training, prefix="ewma5_"
    )
    rbs = (
        rbs.merge(def_df[opp_keep], left_on=["season", "week", "opponent_team"], right_on=["season", "week", "def_team"], how="left")
        .merge(env_df, on=["season", "week", "team"], how="left")
    )
    rbs["implied_team_total"] = rbs["implied_team_total"].fillna(22.5)
    rbs["team_spread"] = rbs["team_spread"].fillna(0.0)
    rbs["is_home"] = rbs["is_home"].fillna(0)

    return {"WR_TE": wr_te, "QB": qbs, "RB": rbs}
