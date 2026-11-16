"""
Weekly post-game grading and cumulative season accuracy tracker.

Compares saved pre-kickoff predictions (`data/latest_predictions.csv` and
`data/latest_player_predictions.csv`) against official nflverse final box scores,
prints a weekly & cumulative season scorecard, and appends deduplicated records to:
  - `data/season_game_tracker.csv`
  - `data/season_player_tracker.csv`
"""

import os
import nflreadpy as nfl
import numpy as np
import pandas as pd

GAME_PRED_PATH = "data/latest_predictions.csv"
PLAYER_PRED_PATH = "data/latest_player_predictions.csv"
GAME_TRACKER_PATH = "data/season_game_tracker.csv"
PLAYER_TRACKER_PATH = "data/season_player_tracker.csv"

def _append_deduped(new_df: pd.DataFrame, tracker_path: str, key_cols: list[str]) -> pd.DataFrame:
    """Appends newly graded rows to a persistent CSV tracker without duplicating already-graded games."""
    if os.path.exists(tracker_path):
        existing = pd.read_csv(tracker_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    else:
        combined = new_df.reset_index(drop=True)

    combined.to_csv(tracker_path, index=False)
    return combined

def grade_games(season: int, week: int) -> None:
    """Grades straight-up game winner picks and probability calibration for completed games."""
    if not os.path.exists(GAME_PRED_PATH):
        print(f"Skipping game grading: {GAME_PRED_PATH} not found.")
        return

    preds = pd.read_csv(GAME_PRED_PATH)
    schedules = nfl.load_schedules([season]).to_pandas()
    week_sched = schedules[
        (schedules["season"] == season)
        & (schedules["week"] == week)
        & (schedules["result"].notna())  # Only grade games with a final score
    ].copy()

    if week_sched.empty:
        print(f"\nNo completed final game scores available yet for Season {season}, Week {week}.")
        return

    week_sched["actual_winner"] = np.where(
        week_sched["result"] > 0,
        week_sched["home_team"],
        np.where(week_sched["result"] < 0, week_sched["away_team"], "TIE"),
    )
    week_sched["home_won"] = (week_sched["result"] > 0).astype(int)

    merged = preds.merge(
        week_sched[["home_team", "away_team", "home_score", "away_score", "actual_winner", "home_won"]],
        on=["home_team", "away_team"],
        how="inner",
    )

    if merged.empty:
        print(f"\nSaved predictions in {GAME_PRED_PATH} do not match completed Week {week} games.")
        return

    # Support any probability column naming across v1-v3 CSV formats
    if "home_win_prob" in merged.columns:
        home_prob_01 = merged["home_win_prob"] / 100.0
    elif "home_prob" in merged.columns:
        home_prob_01 = merged["home_prob"] / 100.0
    elif "win_confidence" in merged.columns:
        conf_01 = merged["win_confidence"] / 100.0
        home_prob_01 = np.where(
            merged["predicted_winner"] == merged["home_team"], conf_01, 1.0 - conf_01
        )
    else:
        home_prob_01 = np.where(merged["predicted_winner"] == merged["home_team"], 0.6, 0.4)

    merged["season"] = season
    merged["week"] = week
    merged["home_win_prob"] = np.round(home_prob_01 * 100.0, 1)
    merged["correct"] = (merged["predicted_winner"] == merged["actual_winner"]).astype(int)
    merged["brier"] = (home_prob_01 - merged["home_won"]) ** 2

    tracker = _append_deduped(
        merged, GAME_TRACKER_PATH, key_cols=["season", "week", "home_team", "away_team"]
    )

    print(f"\n=== GAME PREDICTIONS SCORECARD (Season {season}, Week {week}) ===")
    display_cols = [
        c for c in ["away_team", "home_team", "predicted_winner", "win_confidence", "actual_winner", "correct"]
        if c in merged.columns
    ]
    print(merged[display_cols].to_string(index=False))
    print(
        f"Week {week} Record: {merged['correct'].sum()}-{len(merged) - merged['correct'].sum()} "
        f"({merged['correct'].mean():.1%}) | Brier: {merged['brier'].mean():.4f}"
    )
    print(
        f"Season Cumulative: {tracker['correct'].sum()}-{len(tracker) - tracker['correct'].sum()} "
        f"({tracker['correct'].mean():.1%}) across {len(tracker)} graded games"
    )

def grade_players(season: int, week: int) -> None:
    """Grades player yardage projections, 75th-pct ceilings, and Anytime TD probabilities."""
    if not os.path.exists(PLAYER_PRED_PATH):
        print(f"Skipping player grading: {PLAYER_PRED_PATH} not found.")
        return

    preds = pd.read_csv(PLAYER_PRED_PATH)
    pstats = nfl.load_player_stats([season]).to_pandas()
    week_stats = pstats[(pstats["season"] == season) & (pstats["week"] == week)].copy()

    if week_stats.empty:
        print(f"\nNo player box scores published yet for Season {season}, Week {week}.")
        return

    for col in ["receiving_yards", "passing_yards", "rushing_yards", "receiving_tds", "rushing_tds"]:
        week_stats[col] = week_stats[col].fillna(0.0)

    week_stats["actual_tds"] = week_stats["receiving_tds"] + week_stats["rushing_tds"]
    week_stats["scored_td"] = (week_stats["actual_tds"] >= 1).astype(int)

    merged = preds.merge(
        week_stats[
            [
                "player_display_name",
                "team",
                "receiving_yards",
                "passing_yards",
                "rushing_yards",
                "actual_tds",
                "scored_td",
            ]
        ],
        on=["player_display_name", "team"],
        how="inner",
    )

    if merged.empty:
        print(f"\nSaved player predictions in {PLAYER_PRED_PATH} do not match Week {week} box scores yet.")
        return

    # Map actual yards to the corresponding projected stat category
    merged["actual_yards"] = np.select(
        [
            merged["stat_type"] == "Rec Yards",
            merged["stat_type"] == "Pass Yards",
            merged["stat_type"] == "Rush Yards",
        ],
        [
            merged["receiving_yards"],
            merged["passing_yards"],
            merged["rushing_yards"],
        ],
        default=np.nan,
    )

    merged["season"] = season
    merged["week"] = week
    merged["abs_error"] = (merged["actual_yards"] - merged["proj_yards"]).abs()
    merged["hit_ceiling_75"] = (merged["actual_yards"] >= merged["ceiling_75"]).astype(int)

    keep_cols = [
        c for c in [
            "season",
            "week",
            "player_display_name",
            "position",
            "team",
            "opponent",
            "stat_type",
            "proj_yards",
            "ceiling_75",
            "actual_yards",
            "abs_error",
            "hit_ceiling_75",
            "td_prob",
            "fair_td_odds",
            "scored_td",
        ]
        if c in merged.columns
    ]
    graded = merged[keep_cols].copy()

    tracker = _append_deduped(
        graded, PLAYER_TRACKER_PATH, key_cols=["season", "week", "player_display_name", "team", "stat_type"]
    )

    print(f"\n=== PLAYER PROJECTIONS SCORECARD (Season {season}, Week {week}) ===")
    for stat, grp in graded.groupby("stat_type"):
        print(
            f"  {stat:<10} | Graded: {len(grp):<3} | MAE: {grp['abs_error'].mean():.1f} yds | "
            f"75th-Pct Ceiling Hit Rate: {grp['hit_ceiling_75'].mean():.1%}"
        )

    if "td_prob" in graded.columns:
        td_graded = graded.dropna(subset=["td_prob"]).drop_duplicates(subset=["player_display_name", "team"])
        if not td_graded.empty:
            print(
                f"  Anytime TD | Graded: {len(td_graded):<3} | Avg Predicted: {td_graded['td_prob'].mean():.1f}% | "
                f"Actual TD Rate: {td_graded['scored_td'].mean():.1%}"
            )

    print(f"\nCumulative Graded Player Props in {PLAYER_TRACKER_PATH}: {len(tracker)} rows")

def run_evaluation() -> None:
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Evaluating completed results for Season {current_season}, Week {current_week}...")
    grade_games(current_season, current_week)
    grade_players(current_season, current_week)

if __name__ == "__main__":
    run_evaluation()
