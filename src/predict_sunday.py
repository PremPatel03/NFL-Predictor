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
    prints the weekly matchup table, and writes `data/latest_predictions.csv`.
    """
    # 1. Load the trained v3 model artifact and its exact ordered feature list
    artifact = joblib.load("models/nfl_win_prob_v3.joblib")
    model = artifact["model"]
    feature_cols = artifact["features"]

    # Load the pre-game feature snapshot built earlier by `build_features.py`
    slate = pd.read_parquet("data/pregame_features.parquet")

    if slate.empty:
        print("No games found in data/pregame_features.parquet for the active week.")
        return slate

    # Query ESPN for live Sunday morning starting QB inactive status
    qb_outs = get_live_qb_outs(slate)
    slate["home_qb_out"] = slate["home_team"].map(qb_outs).fillna(0).astype(int)
    slate["away_qb_out"] = slate["away_team"].map(qb_outs).fillna(0).astype(int)

    # 4. Predict calibrated home win probabilities (class 1 = home_win)
    X_live = slate[feature_cols].fillna(0.0)
    home_probs = model.predict_proba(X_live)[:, 1]

    slate["home_win_prob"] = (home_probs * 100).round(1)
    slate["away_win_prob"] = ((1.0 - home_probs) * 100).round(1)
    slate["predicted_winner"] = np.where(
        home_probs >= 0.5, slate["home_team"], slate["away_team"]
    )
    slate["win_confidence"] = np.maximum(slate["home_win_prob"], slate["away_win_prob"])

    pd.set_option("display.width", 1000)
    # Format clean output table sorted by highest model confidence
    output_cols = [
        "gameday",
        "away_team",
        "home_team",
        "predicted_winner",
        "win_confidence",
        "home_win_prob",
        "away_win_prob",
    ]
    available_cols = [c for c in output_cols if c in slate.columns]
    results = slate[available_cols].sort_values("win_confidence", ascending=False).reset_index(drop=True)

    if "elo_diff" in slate.columns:
        slate["elo_diff"] = slate["elo_diff"].round(1)

    # Save to CSV so GitHub Actions automatically commits the updated picks to the repo
    results.to_csv("data/latest_predictions.csv", index=False)

    print("\n=== WEEKLY NFL PREDICTIONS (STARTER-FILTERED) ===")
    print(results.to_string(index=False))
    print("\nSaved predictions to data/latest_predictions.csv")
    return results

if __name__ == "__main__":
    run_sunday_predictions()
