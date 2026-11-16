import nflreadpy as nfl
import pandas as pd

def build_weekly_snapshot() -> pd.DataFrame:
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Building feature snapshot for Season {current_season}, Week {current_week}...")

    # Load current season (and prior season so Week 1-4 rolling averages have history)
    seasons = [current_season - 1, current_season]
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()

    # Calculate game-level offensive & defensive EPA from completed games
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    off_epa = (
        plays.groupby(["season", "week", "posteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "off_pass_epa", "run": "off_run_epa"})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_epa = (
        plays.groupby(["season", "week", "defteam", "play_type"])["epa"]
        .mean()
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_run_epa"})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    team_stats = pd.merge(off_epa, def_epa, on=["season", "week", "team"])
    team_stats = team_stats.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Compute 5-game EWMA (NO .shift(1) here because we want each team's latest completed game)
    epa_cols = ["off_pass_epa", "off_run_epa", "def_pass_epa", "def_run_epa"]
    for col in epa_cols:
        team_stats[f"ewma_{col}"] = (
            team_stats.groupby("team")[col]
            .transform(lambda s: s.ewm(span=5, min_periods=1).mean())
        )

    # Grab each team's most recent completed game form heading into this week
    latest_form = team_stats.groupby("team").tail(1)[
        ["team"] + [f"ewma_{c}" for c in epa_cols]
    ]

    # Filter schedule to the target week
    slate = schedules[
        (schedules["season"] == current_season) & (schedules["week"] == current_week)
    ].copy()

    slate = slate.merge(
        latest_form.add_prefix("home_"),
        left_on="home_team", right_on="home_team", how="left"
    ).merge(
        latest_form.add_prefix("away_"),
        left_on="away_team", right_on="away_team", how="left"
    )

    # Compute exact same differentials used during training
    slate["pass_epa_diff"] = slate["home_ewma_off_pass_epa"] - slate["away_ewma_off_pass_epa"]
    slate["run_epa_diff"] = slate["home_ewma_off_run_epa"] - slate["away_ewma_off_run_epa"]
    slate["def_pass_epa_diff"] = slate["away_ewma_def_pass_epa"] - slate["home_ewma_def_pass_epa"]
    slate["def_run_epa_diff"] = slate["away_ewma_def_run_epa"] - slate["home_ewma_def_run_epa"]
    slate["rest_diff"] = slate["home_rest"] - slate["away_rest"]

    output_path = "data/pregame_features.parquet"
    slate.to_parquet(output_path, index=False)
    print(f"Saved {len(slate)} matchups to {output_path}")
    return slate

if __name__ == "__main__":
    build_weekly_snapshot()
