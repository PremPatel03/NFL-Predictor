import nflreadpy as nfl
import pandas as pd

from features import add_matchup_differentials, compute_elo_ratings, compute_team_rolling_stats

def build_weekly_snapshot() -> pd.DataFrame:
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Building v3 feature snapshot for Season {current_season}, Week {current_week}...")

    # Pull from 2023 onward so Elo ratings are fully warmed up
    seasons = list(range(2023, current_season + 1))
    stats = compute_team_rolling_stats(seasons, shift_for_training=False)
    schedules = nfl.load_schedules(seasons).to_pandas()
    elo_df = compute_elo_ratings(schedules)

    latest_form = stats.groupby("team").tail(1)
    slate = schedules[
        (schedules["season"] == current_season) & (schedules["week"] == current_week)
    ].copy()

    slate = (
        slate.merge(elo_df[["game_id", "elo_diff"]], on="game_id", how="left")
        .merge(
            latest_form.add_prefix("home_"),
            left_on="home_team", right_on="home_team", how="left",
        )
        .merge(
            latest_form.add_prefix("away_"),
            left_on="away_team", right_on="away_team", how="left",
        )
    )

    slate = add_matchup_differentials(slate)
    slate.to_parquet("data/pregame_features.parquet", index=False)
    print(f"Saved {len(slate)} matchups to data/pregame_features.parquet")
    return slate

if __name__ == "__main__":
    build_weekly_snapshot()
