"""
Weekly NFL player yardage & Anytime Touchdown probability projection script.

Runs on Sunday morning (11:40 AM EST via GitHub Actions) or manually via CLI:
  - `python src/predict_players.py` -> outputs top projections & saves CSV
  - `python src/predict_players.py --player "Zay Flowers"` -> looks up a specific player
"""

import argparse
import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests
import os

from player_features import (
    QB_FEATURES,
    RB_FEATURES,
    TD_FEATURES,
    WR_TE_FEATURES,
    build_all_player_datasets,
    build_td_dataset,
)

# ESPN public scoreboard & game summary endpoints for real-time Sunday morning inactives
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}

def _prob_to_american_odds(prob: float) -> str:
    """Converts a probability (0..1) into fair American odds string (e.g., '+145' or '-120')."""
    p = float(np.clip(prob, 0.01, 0.99))
    if p >= 0.5:
        odds = -round((p / (1.0 - p)) * 100)
        return f"{odds}"
    odds = round(((1.0 - p) / p) * 100)
    return f"+{odds}"

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
    def_join_cols: list[str],
) -> pd.DataFrame:
    """
    Pairs each active player's latest rolling stats with their upcoming opponent's
    defensive metrics and Vegas game environment (spread & implied team total).
    """
    home_rows = slate[["season", "week", "home_team", "away_team", "spread_line", "total_line", "location"]].copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["is_home"] = (home_rows["location"] == "Home").astype(int)
    home_rows["team_spread"] = home_rows["spread_line"].fillna(0.0)
    home_rows["implied_team_total"] = ((home_rows["total_line"] + home_rows["spread_line"]) / 2.0).fillna(22.5)

    away_rows = slate[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["is_home"] = 0
    away_rows["team_spread"] = (-away_rows["spread_line"]).fillna(0.0)
    away_rows["implied_team_total"] = ((away_rows["total_line"] - away_rows["spread_line"]) / 2.0).fillna(22.5)

    team_matchups = pd.concat(
        [
            home_rows[["season", "week", "team", "opponent", "is_home", "team_spread", "implied_team_total"]],
            away_rows[["season", "week", "team", "opponent", "is_home", "team_spread", "implied_team_total"]],
        ],
        ignore_index=True,
    )

    merged = latest_players.merge(team_matchups, on="team", how="inner")
    merged = merged.merge(latest_def[["def_team"] + def_join_cols], left_on="opponent", right_on="def_team", how="left")
    merged = merged[~merged["player_display_name"].str.strip().str.lower().isin(inactive_names)].copy()
    return merged

def run_player_predictions(player_query: str | None = None, team_query: str | None = None):
    """
    Generates weekly yardage projections, 75th-percentile ceilings, and Anytime TD
    probabilities + fair odds across WR, TE, QB, and RB positions.
    """
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    print(f"Generating player yardage & TD projections for Season {current_season}, Week {current_week}...")

    seasons = [current_season - 1, current_season]
    datasets = build_all_player_datasets(seasons, shift_for_training=False)
    td_df = build_td_dataset(seasons, shift_for_training=False)

    schedules = nfl.load_schedules(seasons).to_pandas()
    slate = schedules[(schedules["season"] == current_season) & (schedules["week"] == current_week)].copy()
    inactive_names = get_inactive_players()

    """ # Load official current-season rosters so only ACTIVE players on their CURRENT team appear
    rosters = nfl.load_rosters([current_season]).to_pandas()
    active_roster = rosters[rosters["status"] == "ACT"][["gsis_id", "team"]].drop_duplicates(subset=["gsis_id"])
    active_roster = active_roster.rename(columns={"gsis_id": "player_id", "team": "current_team"}) """

    def _filter_active_recent(df: pd.DataFrame) -> pd.DataFrame:
        latest = df.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(1).copy()
        # Keep only players who actually logged stats in the current season within the last 3 weeks
        return latest[(latest["season"] == current_season) & (latest["week"] >= current_week - 3)]

    # Latest defensive generosity snapshots
    yard_def_cols = [
        "opp_ewma8_def_pass_epa",
        "opp_ewma8_def_run_epa",
        "opp_ewma8_wr_te_yards_allowed",
        "opp_ewma8_pass_yards_allowed",
        "opp_ewma8_rush_yards_allowed",
    ]
    latest_yard_def = (
        datasets["WR_TE"]
        .dropna(subset=["opponent_team"])
        .sort_values(["season", "week"])
        .groupby("opponent_team")
        .tail(1)[["opponent_team"] + yard_def_cols]
        .rename(columns={"opponent_team": "def_team"})
    )

    latest_td_def = (
        td_df.dropna(subset=["opponent_team"])
        .sort_values(["season", "week"])
        .groupby("opponent_team")
        .tail(1)[["opponent_team", "opp_ewma8_tds_allowed"]]
        .rename(columns={"opponent_team": "def_team"})
    )

    all_preds = []

    # 1. WR & TE Receiving Yards
    wr_latest = _filter_active_recent(datasets["WR_TE"])
    wr_latest = wr_latest[wr_latest["ewma5_targets"] >= 2.0].copy()
    wr_keep = [
        "player_id", "player_display_name", "position", "team", "is_wr",
        "ewma5_rec_yards", "ewma5_targets", "ewma5_receptions",
        "ewma5_target_share", "ewma5_air_yards_share", "ewma5_rec_epa",
    ]
    wr_slate = _build_upcoming_slate_rows(wr_latest[wr_keep], latest_yard_def, slate, inactive_names, yard_def_cols).dropna(subset=WR_TE_FEATURES)

    wr_art = joblib.load("models/wr_te_yards_v1.joblib")
    wr_slate["proj_yards"] = np.clip(wr_art["model"].predict(wr_slate[WR_TE_FEATURES]), 0, None)
    wr_slate["ceiling_75"] = wr_slate["proj_yards"] + wr_art["upside_75_boost"]
    wr_slate["stat_type"] = "Rec Yards"
    all_preds.append(wr_slate[["player_id", "player_display_name", "position", "team", "opponent", "stat_type", "proj_yards", "ceiling_75"]])

    # 2. QB Passing Yards
    qb_latest = _filter_active_recent(datasets["QB"])
    qb_latest = qb_latest[qb_latest["ewma5_attempts"] >= 15.0].copy()
    qb_keep = [
        "player_id", "player_display_name", "position", "team",
        "ewma5_pass_yards", "ewma5_attempts", "ewma5_completions",
        "ewma5_pass_epa", "ewma5_yards_per_attempt",
    ]
    qb_slate = _build_upcoming_slate_rows(qb_latest[qb_keep], latest_yard_def, slate, inactive_names, yard_def_cols).dropna(subset=QB_FEATURES)

    qb_art = joblib.load("models/qb_pass_yards_v1.joblib")
    qb_slate["proj_yards"] = np.clip(qb_art["model"].predict(qb_slate[QB_FEATURES]), 0, None)
    qb_slate["ceiling_75"] = qb_slate["proj_yards"] + qb_art["upside_75_boost"]
    qb_slate["stat_type"] = "Pass Yards"
    all_preds.append(qb_slate[["player_id", "player_display_name", "position", "team", "opponent", "stat_type", "proj_yards", "ceiling_75"]])

    # 3. RB Rushing Yards
    rb_latest = _filter_active_recent(datasets["RB"])
    rb_latest = rb_latest[rb_latest["ewma5_carries"] >= 5.0].copy()
    rb_keep = [
        "player_id", "player_display_name", "position", "team",
        "ewma5_rush_yards", "ewma5_carries", "ewma5_rush_epa",
        "ewma5_yards_per_carry", "ewma5_targets",
    ]
    rb_slate = _build_upcoming_slate_rows(rb_latest[rb_keep], latest_yard_def, slate, inactive_names, yard_def_cols).dropna(subset=RB_FEATURES)

    rb_art = joblib.load("models/rb_rush_yards_v1.joblib")
    rb_slate["proj_yards"] = np.clip(rb_art["model"].predict(rb_slate[RB_FEATURES]), 0, None)
    rb_slate["ceiling_75"] = rb_slate["proj_yards"] + rb_art["upside_75_boost"]
    rb_slate["stat_type"] = "Rush Yards"
    all_preds.append(rb_slate[["player_id", "player_display_name", "position", "team", "opponent", "stat_type", "proj_yards", "ceiling_75"]])

    # 4. Anytime Touchdown Probabilities (WR, TE, RB)
    td_latest = _filter_active_recent(td_df)
    td_latest = td_latest[td_latest["ewma5_total_touches"] >= 2.5].copy()
    td_keep = [
        "player_id", "player_display_name", "position", "team", "is_rb",
        "ewma5_rz_opps", "ewma5_gl_opps", "ewma5_total_tds",
        "ewma5_total_touches", "ewma5_total_yards",
    ]
    td_slate = _build_upcoming_slate_rows(td_latest[td_keep], latest_td_def, slate, inactive_names, ["opp_ewma8_tds_allowed"]).dropna(subset=TD_FEATURES)

    td_art = joblib.load("models/anytime_td_v1.joblib")
    td_probs = td_art["model"].predict_proba(td_slate[TD_FEATURES])[:, 1]
    td_slate["td_prob"] = (td_probs * 100).round(1)
    td_slate["fair_td_odds"] = [_prob_to_american_odds(p) for p in td_probs]

    # 5. Merge Anytime TD Probabilities onto Yardage Projections
    results = pd.concat(all_preds, ignore_index=True)
    results = results.merge(
        td_slate[["player_id", "td_prob", "fair_td_odds"]],
        on="player_id",
        how="left",
    ).drop(columns=["player_id"])

    results["proj_yards"] = results["proj_yards"].round(1)
    results["ceiling_75"] = results["ceiling_75"].round(1)
    results = results.sort_values(["stat_type", "proj_yards"], ascending=[True, False])

    os.makedirs("data/history", exist_ok=True)
    results.to_csv("data/latest_player_predictions.csv", index=False)
    results.to_csv(f"data/history/player_preds_{current_season}_w{current_week}.csv", index=False)

    if player_query:
        matches = results[results["player_display_name"].str.contains(player_query, case=False, na=False)]
        print(f"\n=== PLAYER LOOKUP: '{player_query}' ===")
        print(matches.to_string(index=False) if not matches.empty else "No matching active player found on this week's slate.")
    elif team_query:
        team_code = team_query.strip().upper()
        game_rows = results[(results["team"] == team_code) | (results["opponent"] == team_code)].copy()
        game_rows = game_rows.sort_values(["team", "stat_type", "proj_yards"], ascending=[True, True, False])
        print(f"\n=== FULL GAME OFFENSIVE PROJECTIONS ({team_code} MATCHUP) ===")
        print(game_rows.to_string(index=False) if not game_rows.empty else f"No active matchup found for '{team_code}'.")
    else:
        print("\n=== TOP 10 PROJECTED RECEIVERS THIS WEEK ===")
        print(results[results["stat_type"] == "Rec Yards"].head(10).to_string(index=False))
        print("\n=== TOP 10 PROJECTED RUSHERS THIS WEEK ===")
        print(results[results["stat_type"] == "Rush Yards"].head(10).to_string(index=False))
        print("\n=== TOP 10 ANYTIME TOUCHDOWN CANDIDATES THIS WEEK ===")
        top_td = results.dropna(subset=["td_prob"]).drop_duplicates(subset=["player_display_name", "team"]).sort_values("td_prob", ascending=False)
        print(top_td[["player_display_name", "position", "team", "opponent", "td_prob", "fair_td_odds"]].head(10).to_string(index=False))
        print("\nSaved full projections to data/latest_player_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly NFL player yardage & TD projections.")
    parser.add_argument("--player", type=str, default=None, help="Filter output to a specific player name")
    parser.add_argument("--team", type=str, default=None, help="Show full offensive projections for a team's game (e.g., KC)")
    args = parser.parse_args()
    run_player_predictions(args.player, args.team)
