"""
Weekly post-game grading and cumulative season accuracy tracker.

Scans archived pre-kickoff prediction files in `data/history/` (and `data/latest_*.csv`),
matches them against official nflverse final box scores, prints a weekly & cumulative
season scorecard, and appends deduplicated records to:
  - `data/season_game_tracker.csv`
  - `data/season_player_tracker.csv`
"""

import glob
import os
import re
import nflreadpy as nfl
import numpy as np
import pandas as pd

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

def grade_all_games(season: int) -> None:
    """Grades straight-up game winner picks and Brier scores for all archived weeks in `season`."""
    schedules = nfl.load_schedules([season]).to_pandas()
    completed = schedules[(schedules["season"] == season) & (schedules["result"].notna())].copy()

    if completed.empty:
        print(f"No completed final game scores available yet for Season {season}.")
        return

    completed["actual_winner"] = np.where(
        completed["result"] > 0,
        completed["home_team"],
        np.where(completed["result"] < 0, completed["away_team"], "TIE"),
    )
    completed["home_won"] = (completed["result"] > 0).astype(int)

    # Collect archived weekly prediction files + fallback to latest_predictions.csv
    pred_files = sorted(glob.glob(f"data/history/game_preds_{season}_w*.csv"))
    if not pred_files and os.path.exists("data/latest_predictions.csv"):
        pred_files = ["data/latest_predictions.csv"]

    if not pred_files:
        print("No saved game prediction files found to grade.")
        return

    all_graded = []
    for path in pred_files:
        preds = pd.read_csv(path)
        match = re.search(r"_w(\d+)\.csv$", path)
        file_week = int(match.group(1)) if match else int(nfl.get_current_week())

        week_sched = completed[completed["week"] == file_week]
        if week_sched.empty:
            continue

        merged = preds.merge(
            week_sched[["week", "home_team", "away_team", "home_score", "away_score", "actual_winner", "home_won"]],
            on=["home_team", "away_team"],
            how="inner",
        )
        if merged.empty:
            continue

        if "home_win_prob" in merged.columns:
            home_prob_01 = merged["home_win_prob"] / 100.0
        elif "win_confidence" in merged.columns:
            conf_01 = merged["win_confidence"] / 100.0
            home_prob_01 = np.where(merged["predicted_winner"] == merged["home_team"], conf_01, 1.0 - conf_01)
        else:
            home_prob_01 = np.where(merged["predicted_winner"] == merged["home_team"], 0.6, 0.4)

        merged["season"] = season
        merged["home_win_prob"] = np.round(home_prob_01 * 100.0, 1)
        merged["correct"] = (merged["predicted_winner"] == merged["actual_winner"]).astype(int)
        merged["brier"] = (home_prob_01 - merged["home_won"]) ** 2
        all_graded.append(merged)

    if not all_graded:
        print("No completed games matched archived prediction files yet.")
        return

    graded_df = pd.concat(all_graded, ignore_index=True)
    tracker = _append_deduped(
        graded_df, GAME_TRACKER_PATH, key_cols=["season", "week", "home_team", "away_team"]
    )

    latest_week = int(graded_df["week"].max())
    latest_wk_df = graded_df[graded_df["week"] == latest_week]

    print(f"\n=== GAME PREDICTIONS SCORECARD (Season {season}, Week {latest_week}) ===")
    display_cols = [
        c for c in ["away_team", "home_team", "predicted_winner", "win_confidence", "actual_winner", "correct"]
        if c in latest_wk_df.columns
    ]
    print(latest_wk_df[display_cols].to_string(index=False))
    print(
        f"Week {latest_week} Record: {latest_wk_df['correct'].sum()}-{len(latest_wk_df) - latest_wk_df['correct'].sum()} "
        f"({latest_wk_df['correct'].mean():.1%}) | Brier: {latest_wk_df['brier'].mean():.4f}"
    )
    print(
        f"Season Cumulative: {tracker['correct'].sum()}-{len(tracker) - tracker['correct'].sum()} "
        f"({tracker['correct'].mean():.1%}) across {len(tracker)} graded games"
    )

def grade_all_players(season: int) -> None:
    """Grades player yardage projections, 75th-pct ceilings, and Anytime TD probabilities."""
    pstats = nfl.load_player_stats([season]).to_pandas()
    reg_stats = pstats[pstats["season_type"] == "REG"].copy()

    if reg_stats.empty:
        print(f"No player box scores published yet for Season {season}.")
        return

    for col in ["receiving_yards", "passing_yards", "rushing_yards", "receiving_tds", "rushing_tds"]:
        reg_stats[col] = reg_stats[col].fillna(0.0)

    reg_stats["actual_tds"] = reg_stats["receiving_tds"] + reg_stats["rushing_tds"]
    reg_stats["scored_td"] = (reg_stats["actual_tds"] >= 1).astype(int)

    pred_files = sorted(glob.glob(f"data/history/player_preds_{season}_w*.csv"))
    if not pred_files and os.path.exists("data/latest_player_predictions.csv"):
        pred_files = ["data/latest_player_predictions.csv"]

    if not pred_files:
        print("No saved player prediction files found to grade.")
        return

    all_graded = []
    for path in pred_files:
        preds = pd.read_csv(path)
        match = re.search(r"_w(\d+)\.csv$", path)
        file_week = int(match.group(1)) if match else int(nfl.get_current_week())

        week_stats = reg_stats[reg_stats["week"] == file_week]
        if week_stats.empty:
            continue

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
            continue

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
        merged["week"] = file_week
        merged["abs_error"] = (merged["actual_yards"] - merged["proj_yards"]).abs()
        merged["hit_ceiling_75"] = (merged["actual_yards"] >= merged["ceiling_75"]).astype(int)
        all_graded.append(merged)

    if not all_graded:
        print("No completed player box scores matched archived prediction files yet.")
        return

    combined = pd.concat(all_graded, ignore_index=True)
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
        if c in combined.columns
    ]
    graded = combined[keep_cols].copy()
    tracker = _append_deduped(
        graded, PLAYER_TRACKER_PATH, key_cols=["season", "week", "player_display_name", "team", "stat_type"]
    )

    latest_week = int(graded["week"].max())
    latest_wk = graded[graded["week"] == latest_week]

    print(f"\n=== PLAYER PROJECTIONS SCORECARD (Season {season}, Week {latest_week}) ===")
    for stat, grp in latest_wk.groupby("stat_type"):
        print(
            f"  {stat:<10} | Graded: {len(grp):<3} | MAE: {grp['abs_error'].mean():.1f} yds | "
            f"75th-Pct Ceiling Hit Rate: {grp['hit_ceiling_75'].mean():.1%}"
        )

    if "td_prob" in latest_wk.columns:
        td_graded = latest_wk.dropna(subset=["td_prob"]).drop_duplicates(subset=["player_display_name", "team"])
        if not td_graded.empty:
            print(
                f"  Anytime TD | Graded: {len(td_graded):<3} | Avg Predicted: {td_graded['td_prob'].mean():.1f}% | "
                f"Actual TD Rate: {td_graded['scored_td'].mean():.1%}"
            )

    print(f"\nCumulative Graded Player Props in {PLAYER_TRACKER_PATH}: {len(tracker)} rows")

def run_evaluation() -> None:
    current_season = nfl.get_current_season()
    print(f"Evaluating completed results for Season {current_season}...")
    grade_all_games(current_season)
    grade_all_players(current_season)

if __name__ == "__main__":
    run_evaluation()
