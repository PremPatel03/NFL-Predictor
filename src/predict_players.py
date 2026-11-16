"""
Weekly NFL player yardage projection script (WR/TE Receiving, QB Passing, RB Rushing).

Runs on Sunday morning (11:40 AM EST via GitHub Actions) or manually via CLI:
  - `python src/predict_players.py` -> outputs top league-wide projections & saves CSV
  - `python src/predict_players.py --player "Zay Flowers"` -> looks up a specific player
"""

import argparse
import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests

from player_features import (
    QB_FEATURES,
    RB_FEATURES,
    WR_TE_FEATURES,
    build_all_player_datasets,
)

# ESPN public scoreboard & game summary endpoints used to pull real-time Sunday morning inactives
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}

def get_inactive_players() -> set[str]:
    """
    Queries ESPN's live game summaries for the current slate and returns a lowercase set
    of player display names officially ruled Out, Inactive, or on Injured Reserve.
    """
    sb = requests.get(SCOREBOARD_URL, timeout=15).json()
    out_names = set()

    for event in sb.get("events", []):
        summary = requests.get(SUMMARY_URL, params={"event": event["id"]}, timeout=15).json()
        for team_block in summary.get("injuries", []):
            for item in team_block.get("injuries", []):
                status = item.get("status", "").lower()
                if status in {"out", "inactive", "injured reserve"}:
                    name = item.get("athlete", {}).get("displayName", "").strip().lower()
                    if name:
                        out_names.add(name)
    return out_names

def _build_upcoming_slate_rows(
    latest_players: pd.DataFrame,
    latest_def: pd.DataFrame,
    slate: pd.DataFrame,
    inactive_names: set[str],
) -> pd.DataFrame:
    """
    Pairs each active player's most recent 5-game rolling stats with their upcoming
    opponent's defensive metrics and Vegas game environment (spread & implied team total).
    """
    # Build home-team matchup rows (in nflverse, positive spread_line = home team favored)
    home_rows = slate[["season", "week", "home_team", "away_team", "spread_line", "total_line", "location"]].copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["is_home"] = (home_rows["location"] == "Home").astype(int)
    home_rows["team_spread"] = home_rows["spread_line"].fillna(0.0)
    home_rows["implied_team_total"] = ((home_rows["total_line"] + home_rows["spread_line"]) / 2.0).fillna(22.5)

    # Build away-team matchup rows (flip spread sign so positive always = favored)
    away_rows = slate[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["is_home"] = 0
    away_rows["team_spread"] = (-away_rows["spread_line"]).fillna(0.0)
    away_rows["implied_team_total"] = ((away_rows["total_line"] - away_rows["spread_line"]) / 2.0).fillna(22.5)

    # Combine into one lookup table keyed by team abbreviation
    team_matchups = pd.concat(
        [
            home_rows[["season", "week", "team", "opponent", "is_home", "team_spread", "implied_team_total"]],
            away_rows[["season", "week", "team", "opponent", "is_home", "team_spread", "implied_team_total"]],
        ],
        ignore_index=True,
    )

    # Inner-join players onto this week's active schedule (automatically drops teams on a BYE week)
    merged = latest_players.merge(team_matchups, on="team", how="inner")

    # Attach the upcoming opponent defense's 8-game rolling yards allowed & defensive EPA
    merged = merged.merge(latest_def, left_on="opponent", right_on="def_team", how="left")

    # Exclude any player ruled Out/Inactive on ESPN's live injury feed
    merged = merged[~merged["player_display_name"].str.strip().str.lower().isin(inactive_names)].copy()
    return merged

def run_player_predictions(player_query: str | None = None):
    """
    Loads the three trained position Ridge models (`WR_TE`, `QB`, `RB`), generates
    baseline yardage projections + 75th-percentile ceilings for the current NFL week,
    saves all projections to `data/latest_player_predictions.csv`, and prints results.
    """
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Generating player yardage projections for Season {current_season}, Week {current_week}...")

    # Pull prior + current season so early-week rolling windows carry smooth historical priors
    seasons = [current_season - 1, current_season]

    # shift_for_training=False so all completed games through last week inform today's projections
    datasets = build_all_player_datasets(seasons, shift_for_training=False)
    schedules = nfl.load_schedules(seasons).to_pandas()
    slate = schedules[(schedules["season"] == current_season) & (schedules["week"] == current_week)].copy()

    # Fetch live Sunday morning inactive names from ESPN
    inactive_names = get_inactive_players()

    # Extract each defense's single most recent 8-game rolling generosity snapshot
    opp_cols = [
        "opp_ewma8_def_pass_epa",
        "opp_ewma8_def_run_epa",
        "opp_ewma8_wr_te_yards_allowed",
        "opp_ewma8_pass_yards_allowed",
        "opp_ewma8_rush_yards_allowed",
    ]
    latest_def = (
        datasets["WR_TE"]
        .dropna(subset=["opponent_team"])
        .sort_values(["season", "week"])
        .groupby("opponent_team")
        .tail(1)[["opponent_team"] + opp_cols]
        .rename(columns={"opponent_team": "def_team"})
    )

    all_preds = []

    # WR & TE Receiving Yards Projections
    wr_latest = (
        datasets["WR_TE"]
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
    )
    # Filter to active rotation pass-catchers averaging at least 2.0 targets/game
    wr_latest = wr_latest[wr_latest["ewma5_targets"] >= 2.0].copy()
    wr_keep = [
        "player_id", "player_display_name", "position", "team", "is_wr",
        "ewma5_rec_yards", "ewma5_targets", "ewma5_receptions",
        "ewma5_target_share", "ewma5_air_yards_share", "ewma5_rec_epa",
    ]
    wr_slate = _build_upcoming_slate_rows(wr_latest[wr_keep], latest_def, slate, inactive_names).dropna(subset=WR_TE_FEATURES)

    wr_art = joblib.load("models/wr_te_yards_v1.joblib")
    wr_slate["proj_yards"] = np.clip(wr_art["model"].predict(wr_slate[WR_TE_FEATURES]), 0, None)
    wr_slate["ceiling_75"] = wr_slate["proj_yards"] + wr_art["upside_75_boost"]
    wr_slate["stat_type"] = "Rec Yards"
    all_preds.append(wr_slate[["player_display_name", "position", "team", "opponent", "stat_type", "proj_yards", "ceiling_75"]])

    # QB Passing Yards Projections
    qb_latest = (
        datasets["QB"]
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
    )
    # Filter to starting-volume QBs averaging at least 15.0 pass attempts/game
    qb_latest = qb_latest[qb_latest["ewma5_attempts"] >= 15.0].copy()
    qb_keep = [
        "player_id", "player_display_name", "position", "team",
        "ewma5_pass_yards", "ewma5_attempts", "ewma5_completions",
        "ewma5_pass_epa", "ewma5_yards_per_attempt",
    ]
    qb_slate = _build_upcoming_slate_rows(qb_latest[qb_keep], latest_def, slate, inactive_names).dropna(subset=QB_FEATURES)

    qb_art = joblib.load("models/qb_pass_yards_v1.joblib")
    qb_slate["proj_yards"] = np.clip(qb_art["model"].predict(qb_slate[QB_FEATURES]), 0, None)
    qb_slate["ceiling_75"] = qb_slate["proj_yards"] + qb_art["upside_75_boost"]
    qb_slate["stat_type"] = "Pass Yards"
    all_preds.append(qb_slate[["player_display_name", "position", "team", "opponent", "stat_type", "proj_yards", "ceiling_75"]])

    # RB Rushing Yards Projections
    rb_latest = (
        datasets["RB"]
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
    )
    # Filter to active backfield contributors averaging at least 5.0 carries/game
    rb_latest = rb_latest[rb_latest["ewma5_carries"] >= 5.0].copy()
    rb_keep = [
        "player_id", "player_display_name", "position", "team",
        "ewma5_rush_yards", "ewma5_carries", "ewma5_rush_epa",
        "ewma5_yards_per_carry", "ewma5_targets",
    ]
    rb_slate = _build_upcoming_slate_rows(rb_latest[rb_keep], latest_def, slate, inactive_names).dropna(subset=RB_FEATURES)

    rb_art = joblib.load("models/rb_rush_yards_v1.joblib")
    rb_slate["proj_yards"] = np.clip(rb_art["model"].predict(rb_slate[RB_FEATURES]), 0, None)
    rb_slate["ceiling_75"] = rb_slate["proj_yards"] + rb_art["upside_75_boost"]
    rb_slate["stat_type"] = "Rush Yards"
    all_preds.append(rb_slate[["player_display_name", "position", "team", "opponent", "stat_type", "proj_yards", "ceiling_75"]])

    #  Combine, Format, Save CSV & Display Results
    results = pd.concat(all_preds, ignore_index=True)
    results["proj_yards"] = results["proj_yards"].round(1)
    results["ceiling_75"] = results["ceiling_75"].round(1)
    results = results.sort_values(["stat_type", "proj_yards"], ascending=[True, False])

    # Save full weekly projections so GitHub Actions can auto-commit them to the repo
    results.to_csv("data/latest_player_predictions.csv", index=False)

    # Either print a single player lookup or the top 10 leaders for each stat category
    if player_query:
        matches = results[results["player_display_name"].str.contains(player_query, case=False, na=False)]
        print(f"\n=== PLAYER LOOKUP: '{player_query}' ===")
        print(matches.to_string(index=False) if not matches.empty else "No matching active player found on this week's slate.")
    else:
        print("\n=== TOP 10 PROJECTED RECEIVERS THIS WEEK ===")
        print(results[results["stat_type"] == "Rec Yards"].head(10).to_string(index=False))
        print("\n=== TOP 10 PROJECTED PASSERS THIS WEEK ===")
        print(results[results["stat_type"] == "Pass Yards"].head(10).to_string(index=False))
        print("\n=== TOP 10 PROJECTED RUSHERS THIS WEEK ===")
        print(results[results["stat_type"] == "Rush Yards"].head(10).to_string(index=False))
        print("\nSaved full projections to data/latest_player_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly NFL player yardage projections.")
    parser.add_argument("--player", type=str, default=None, help="Filter output to a specific player name")
    args = parser.parse_args()
    run_player_predictions(args.player)
