import nflreadpy as nfl
import pandas as pd

from features import add_matchup_differentials, compute_team_rolling_stats

def build_weekly_snapshot() -> pd.DataFrame:
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Building v2 feature snapshot for Season {current_season}, Week {current_week}...")

    seasons = [current_season - 1, current_season]
    stats = compute_team_rolling_stats(seasons, shift_for_training=False)
    schedules = nfl.load_schedules(seasons).to_pandas()

    latest_form = stats.groupby("team").tail(1)
    slate = schedules[
        (schedules["season"] == current_season) & (schedules["week"] == current_week)
    ].copy()

    slate = slate.merge(
        latest_form.add_prefix("home_"),
        left_on="home_team", right_on="home_team", how="left",
    ).merge(
        latest_form.add_prefix("away_"),
        left_on="away_team", right_on="away_team", how="left",
    )

    slate = add_matchup_differentials(slate)
    slate.to_parquet("data/pregame_features.parquet", index=False)
    print(f"Saved {len(slate)} matchups to data/pregame_features.parquet")
    return slate

if __name__ == "__main__":
    build_weekly_snapshot()
