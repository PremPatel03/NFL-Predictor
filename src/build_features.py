"""
Weekly feature snapshot builder for live Sunday game predictions (v3).

Runs on Sunday morning (8:00 AM EST via GitHub Actions) to gather each team's
most up-to-date Elo rating and 16-game rolling EPA metrics, pairs them onto the
upcoming week's matchups, and saves the snapshot to `data/pregame_features.parquet`.
"""

import nflreadpy as nfl
import pandas as pd

from features import add_matchup_differentials, compute_elo_ratings, compute_team_rolling_stats

def build_weekly_snapshot() -> pd.DataFrame:
    """
    Constructs and saves the pre-game feature table for the current NFL week.

    Returns:
        A DataFrame containing one row per scheduled game this week, complete with
        pre-game Elo differentials, rolling efficiency gaps, rest days, and home flags.
    """
    # Automatically detect the active NFL season and week from nflverse
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Building v3 feature snapshot for Season {current_season}, Week {current_week}...")

    # Load data starting from 2023 so multi-season Elo ratings and 16-game EWMAs
    #    have plenty of historical context to stabilize before the current week.
    seasons = list(range(2023, current_season + 1))

    # shift_for_training=False because we WANT to include last week's completed games
    # when projecting this week's unplayed matchups.
    stats = compute_team_rolling_stats(seasons, shift_for_training=False)
    schedules = nfl.load_schedules(seasons).to_pandas()

    # Compute sequential pre-game Elo ratings across all loaded seasons
    elo_df = compute_elo_ratings(schedules)

    # Grab each team's single most recent rolling efficiency row (through last week)
    latest_form = stats.groupby("team").tail(1)

    # Filter the master schedule down to only the upcoming/current week's games
    slate = schedules[
        (schedules["season"] == current_season) & (schedules["week"] == current_week)
    ].copy()

    # Merge pre-game Elo differentials, home team stats, and away team stats onto the slate
    slate = (
        slate.merge(elo_df[["game_id", "elo_diff"]], on="game_id", how="left")
        .merge(
            latest_form.add_prefix("home_"),
            left_on="home_team",
            right_on="home_team",
            how="left",
        )
        .merge(
            latest_form.add_prefix("away_"),
            left_on="away_team",
            right_on="away_team",
            how="left",
        )
    )

    # Compute head-to-head differentials (home minus away) and neutral-site home flags
    slate = add_matchup_differentials(slate)

    # Save to Parquet so `predict_sunday.py` can load it instantly at 11:40 AM EST
    # and combine it with live Sunday morning ESPN inactive reports
    slate.to_parquet("data/pregame_features.parquet", index=False)
    print(f"Saved {len(slate)} matchups to data/pregame_features.parquet")
    return slate

if __name__ == "__main__":
    build_weekly_snapshot()
