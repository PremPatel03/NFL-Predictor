import nflreadpy as nfl
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "elo_diff",
    "pass_epa_diff_16",
    "run_epa_diff_16",
    "def_pass_epa_diff_16",
    "def_run_epa_diff_16",
    "success_rate_diff_16",
    "point_diff_16",
    "rest_diff",
    "is_home",
    "home_qb_out",
    "away_qb_out",
]

def compute_elo_ratings(schedules: pd.DataFrame) -> pd.DataFrame:
    """Computes leak-free pre-game Elo ratings with offseason mean reversion."""
    games = schedules[schedules["game_type"] == "REG"].sort_values(
        ["season", "week", "gameday", "game_id"]
    ).copy()

    elos: dict[str, float] = {}
    current_season = None
    records = []

    K = 20.0
    HFA = 48.0  # ~2.0 points of home-field advantage in Elo points

    for _, row in games.iterrows():
        season = row["season"]
        if season != current_season:
            # Regress 1/3 toward 1500 at the start of each new NFL season
            for team in elos:
                elos[team] = 1500.0 + (2.0 / 3.0) * (elos[team] - 1500.0)
            current_season = season

        home, away = row["home_team"], row["away_team"]
        home_elo = elos.get(home, 1500.0)
        away_elo = elos.get(away, 1500.0)

        hfa_adj = HFA if row["location"] == "Home" else 0.0
        elo_diff = (home_elo + hfa_adj) - away_elo

        records.append({
            "game_id": row["game_id"],
            "home_pre_elo": home_elo,
            "away_pre_elo": away_elo,
            "elo_diff": elo_diff,
        })

        # Update Elo only if game has been played
        if pd.notna(row["result"]):
            expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            actual_home = 1.0 if row["result"] > 0 else (0.5 if row["result"] == 0 else 0.0)
            mov_mult = np.log(abs(row["result"]) + 1.0) * (
                2.2 / ((elo_diff * (1 if actual_home == 1 else -1)) * 0.001 + 2.2)
            )
            shift = K * mov_mult * (actual_home - expected_home)
            elos[home] = home_elo + shift
            elos[away] = away_elo - shift

    return pd.DataFrame(records)

def compute_team_rolling_stats(seasons: list[int], shift_for_training: bool) -> pd.DataFrame:
    """Computes clean, non-garbage-time rolling team metrics across seasons."""
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    plays = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["wp"].between(0.05, 0.95)
    ].copy()
    plays["success"] = (plays["epa"] > 0).astype(float)

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
    defn = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    comp = schedules[schedules["result"].notna()].copy()
    home_scores = comp[["season", "week", "home_team", "result"]].rename(
        columns={"home_team": "team", "result": "point_diff"}
    )
    away_scores = comp[["season", "week", "away_team", "result"]].assign(
        point_diff=lambda d: -d["result"]
    )[["season", "week", "away_team", "point_diff"]].rename(columns={"away_team": "team"})
    margins = pd.concat([home_scores, away_scores], ignore_index=True)

    stats = (
        off.merge(off_sr, on=["season", "week", "team"])
        .merge(defn, on=["season", "week", "team"])
        .merge(margins, on=["season", "week", "team"])
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

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
        else:
            stats[f"ewma16_{col}"] = grouped.transform(
                lambda s: s.ewm(span=16, min_periods=1).mean()
            )

    return stats

def add_matchup_differentials(df: pd.DataFrame) -> pd.DataFrame:
    df["pass_epa_diff_16"] = df["home_ewma16_off_pass_epa"] - df["away_ewma16_off_pass_epa"]
    df["run_epa_diff_16"] = df["home_ewma16_off_run_epa"] - df["away_ewma16_off_run_epa"]
    df["def_pass_epa_diff_16"] = df["away_ewma16_def_pass_epa"] - df["home_ewma16_def_pass_epa"]
    df["def_run_epa_diff_16"] = df["away_ewma16_def_run_epa"] - df["home_ewma16_def_run_epa"]
    df["success_rate_diff_16"] = (
        df["home_ewma16_off_success_rate"] - df["away_ewma16_off_success_rate"]
    )
    df["point_diff_16"] = df["home_ewma16_point_diff"] - df["away_ewma16_point_diff"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["is_home"] = (df["location"] == "Home").astype(int)
    return df
