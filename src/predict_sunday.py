"""
Weekly Sunday morning NFL game win-probability prediction script (v3).

Runs on Sunday morning (11:40 AM EST via GitHub Actions) or manually via CLI:
   Loads the pre-game Elo & rolling EPA snapshot created by `build_features.py`.
   Queries ESPN's live scoreboard and game summary APIs to check if either team's
     projected starting QB has been ruled Out/Inactive before kickoff.
   Runs the trained Logistic Regression pipeline (`models/nfl_win_prob_v3.joblib`)
     to output calibrated win probabilities for every matchup on the slate.
   Saves the results to `data/latest_predictions.csv` so GitHub Actions can auto-commit them.
"""

import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests
import os
from weather import get_game_weather

# ESPN public scoreboard & game summary endpoints for live Sunday morning injury reports
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

# Map ESPN team abbreviations to nflverse standard abbreviations where they differ
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}

def get_live_qb_outs(slate: pd.DataFrame) -> dict[str, int]:
    """
    Checks ESPN's live injury report for each game on the current week's slate
    and returns a dictionary mapping each team abbreviation to `1` if its listed
    starting QB is officially Out/Inactive, or `0` otherwise.

    Why starter-filtered:
        Ignoring backup/3rd-string QB injuries prevents false alarms when a depth QB
        is inactive while the actual QB1 is healthy and starting.
    """
    # Build a lookup of each team's expected starting QB from the nflverse schedule
    expected_starters: dict[str, str] = {}
    for _, row in slate.iterrows():
        if pd.notna(row.get("home_qb_name")):
            expected_starters[row["home_team"]] = str(row["home_qb_name"]).strip().lower()
        if pd.notna(row.get("away_qb_name")):
            expected_starters[row["away_team"]] = str(row["away_qb_name"]).strip().lower()

    qb_out_flags: dict[str, int] = {team: 0 for team in expected_starters}

    try:
        sb = requests.get(SCOREBOARD_URL, timeout=15).json()
    except Exception as exc:
        print(f"Warning: Could not reach ESPN scoreboard API ({exc}). Defaulting QB flags to 0.")
        return qb_out_flags

    for event in sb.get("events", []):
        try:
            summary = requests.get(SUMMARY_URL, params={"event": event["id"]}, timeout=15).json()
        except Exception:
            continue

        for team_block in summary.get("injuries", []):
            raw_abbr = str(team_block.get("team", {}).get("abbreviation") or "")
            team_abbr = ESPN_TO_NFLVERSE.get(raw_abbr, raw_abbr)
            starter_name = expected_starters.get(team_abbr, "")

            if not starter_name:
                continue

            for item in team_block.get("injuries", []):
                pos = item.get("athlete", {}).get("position", {}).get("abbreviation", "")
                status = item.get("status", "").strip().lower()
                player_name = item.get("athlete", {}).get("displayName", "").strip().lower()

                # Flag only if a QB matching the team's expected starter is Out/Inactive
                if pos == "QB" and status in {"out", "inactive", "injured reserve"}:
                    if player_name == starter_name:
                        qb_out_flags[team_abbr] = 1

    return qb_out_flags

def run_sunday_predictions() -> pd.DataFrame:
    """
    Loads the pregame feature snapshot, injects live Sunday QB injury flags,
    computes calibrated win probabilities with `nfl_win_prob_v3.joblib`,
    compares model probabilities against no-vig Vegas moneylines (+EV edge),
    fetches kickoff-hour stadium weather via Open-Meteo,
    prints the weekly matchup table, and writes `data/latest_predictions.csv`.
    """
    artifact = joblib.load("models/nfl_win_prob_v3.joblib")
    model = artifact["model"]
    feature_cols = artifact["features"]

    slate = pd.read_parquet("data/pregame_features.parquet")
    if slate.empty:
        print("No games found in data/pregame_features.parquet for the active week.")
        return slate

    season_val = int(slate["season"].iloc[0])
    week_val = int(slate["week"].iloc[0])

    # Query ESPN for live Sunday morning starting QB inactive status
    qb_outs = get_live_qb_outs(slate)
    slate["home_qb_out"] = slate["home_team"].map(qb_outs).fillna(0).astype(int)
    slate["away_qb_out"] = slate["away_team"].map(qb_outs).fillna(0).astype(int)

    # Predict calibrated home win probabilities
    X_live = slate[feature_cols].fillna(0.0)
    home_probs = model.predict_proba(X_live)[:, 1]

    slate["home_win_prob"] = (home_probs * 100).round(1)
    slate["away_win_prob"] = ((1.0 - home_probs) * 100).round(1)
    slate["predicted_winner"] = np.where(home_probs >= 0.5, slate["home_team"], slate["away_team"])
    slate["win_confidence"] = np.maximum(slate["home_win_prob"], slate["away_win_prob"])

    # Merge Vegas lines, stadium roof, and 24-hour kickoff time from schedule
    sched_lines = nfl.load_schedules([season_val]).to_pandas()[
        ["game_id", "home_moneyline", "away_moneyline", "spread_line", "total_line", "roof", "gametime"]
    ].rename(columns={"gametime": "raw_gametime"})

    for col in ["home_moneyline", "away_moneyline", "spread_line", "total_line", "roof", "raw_gametime"]:
        if col in slate.columns:
            slate = slate.drop(columns=[col])
    slate = slate.merge(sched_lines, on="game_id", how="left")

    # Format 12-hour ET kickoff times & slate slots
    slate["gametime"] = slate["raw_gametime"]
    parsed_slots = slate.apply(_format_kickoff_slot, axis=1)
    slate["gametime"] = [p[0] for p in parsed_slots]
    slate["time_slot"] = [p[1] for p in parsed_slots]

    # Compute No-Vig Vegas Implied Probabilities & +EV Edge
    raw_home = slate["home_moneyline"].apply(_american_to_implied_prob)
    raw_away = slate["away_moneyline"].apply(_american_to_implied_prob)
    vig_sum = raw_home + raw_away

    slate["vegas_home_prob"] = np.where(vig_sum > 0, (raw_home / vig_sum) * 100.0, 50.0).round(1)
    slate["vegas_away_prob"] = np.where(vig_sum > 0, (raw_away / vig_sum) * 100.0, 50.0).round(1)

    home_picked = slate["predicted_winner"] == slate["home_team"]
    slate["pick_ml"] = np.where(home_picked, slate["home_moneyline"], slate["away_moneyline"])
    slate["pick_ml_str"] = slate["pick_ml"].apply(_format_american)
    slate["vegas_pick_prob"] = np.where(home_picked, slate["vegas_home_prob"], slate["vegas_away_prob"])
    slate["model_edge"] = (slate["win_confidence"] - slate["vegas_pick_prob"]).round(1)

    slate["ev_flag"] = np.where(
        slate["pick_ml"].isna(),
        "No Line",
        np.where(slate["model_edge"] >= 4.0, " +EV Value", np.where(slate["model_edge"] > 0, " Slight Edge", "Market Aligned")),
    )

    # 5. Fetch kickoff-hour stadium weather from Open-Meteo
    weather_rows = [
        get_game_weather(
            r["home_team"],
            str(r["gameday"]),
            str(r.get("raw_gametime", "13:00")),
            r.get("roof"),
        )
        for _, r in slate.iterrows()
    ]
    weather_df = pd.DataFrame(weather_rows)
    for col in ["weather_summary", "temp_f", "wind_mph", "weather_alert"]:
        slate[col] = weather_df[col].values

    if "elo_diff" in slate.columns:
        slate["elo_diff"] = slate["elo_diff"].round(1)

    pd.set_option("display.width", 1000)

    output_cols = [
        "gameday",
        "gametime",
        "time_slot",
        "away_team",
        "home_team",
        "spread_line",
        "total_line",
        "predicted_winner",
        "pick_ml_str",
        "pick_ml",
        "win_confidence",
        "vegas_pick_prob",
        "model_edge",
        "ev_flag",
        "weather_summary",
        "temp_f",
        "wind_mph",
        "weather_alert",
        "home_win_prob",
        "away_win_prob",
    ]
    available_cols = [c for c in output_cols if c in slate.columns]
    results = slate[available_cols].sort_values("win_confidence", ascending=False).reset_index(drop=True)

    os.makedirs("data/history", exist_ok=True)
    results.to_csv("data/latest_predictions.csv", index=False)
    results.to_csv(f"data/history/game_preds_{season_val}_w{week_val}.csv", index=False)

    print("\n=== WEEKLY NFL PREDICTIONS (VEGAS +EV & STADIUM WEATHER) ===")
    print(results.to_string(index=False))
    print("\nSaved predictions to data/latest_predictions.csv")
    return results

def _format_kickoff_slot(row: pd.Series) -> tuple[str, str]:
    """Converts nflverse gameday (YYYY-MM-DD) & gametime (HH:MM ET) into 12h ET time and slate slot."""
    gday = pd.to_datetime(row.get("gameday"), errors="coerce") # type: ignore
    gtime = str(row.get("gametime") or "").strip()

    if pd.isna(gday) or ":" not in gtime:
        return (gtime or "TBD", "Other")

    hour, minute = map(int, gtime.split(":")[:2])
    ampm = "AM" if hour < 12 else "PM"
    hour_12 = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
    formatted_time = f"{hour_12}:{minute:02d} {ampm} ET"

    dow = gday.day_name()
    if dow == "Thursday":
        slot = "Thu Night"
    elif dow == "Monday":
        slot = "Mon Night"
    elif dow == "Saturday":
        slot = "Sat Slate"
    elif dow == "Sunday":
        if hour < 12:
            slot = "Sun International"
        elif hour < 15:
            slot = "Sun 1:00 PM"
        elif hour < 19:
            slot = "Sun 4:00 PM"
        else:
            slot = "Sun Night"
    else:
        slot = dow

    return (formatted_time, slot)

def _american_to_implied_prob(odds: float) -> float:
    """Converts American moneyline odds (e.g., -150 or +130) into raw implied probability (0..1)."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)

def _format_american(odds: float) -> str:
    """Formats numeric American odds as a clean string ('+135' or '-150')."""
    if pd.isna(odds) or odds == 0:
        return "N/A"
    val = int(round(odds))
    return f"+{val}" if val > 0 else f"{val}"

if __name__ == "__main__":
    run_sunday_predictions()
