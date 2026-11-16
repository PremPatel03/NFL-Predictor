import joblib
import pandas as pd
import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

ESPN_TO_NFLVERSE = {
    "WSH": "WAS",
    "LAR": "LA",
}

def get_injured_qbs_by_team() -> dict[str, set[str]]:
    """Returns a mapping of team_abbr -> set of lowercase QB names listed Out/IR."""
    print("Checking live ESPN injury reports...")
    sb = requests.get(SCOREBOARD_URL, timeout=15).json()
    team_out_qbs: dict[str, set[str]] = {}

    for event in sb.get("events", []):
        event_id = event["id"]
        summary = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=15).json()

        for team_block in summary.get("injuries", []):
            raw_abbr = team_block["team"]["abbreviation"]
            team_abbr = ESPN_TO_NFLVERSE.get(raw_abbr, raw_abbr)
            out_names = set()

            for item in team_block.get("injuries", []):
                status = item.get("status", "").lower()
                athlete = item.get("athlete", {})
                pos = athlete.get("position", {}).get("abbreviation", "")
                name = athlete.get("displayName", "").strip().lower()

                if pos == "QB" and status in {"out", "inactive", "injured reserve"}:
                    out_names.add(name)

            team_out_qbs[team_abbr] = out_names

    return team_out_qbs

def is_starter_out(team_abbr: str, starter_name: str, out_map: dict[str, set[str]]) -> int:
    if not isinstance(starter_name, str) or not starter_name.strip():
        return 0
    injured_qbs = out_map.get(team_abbr, set())
    return int(starter_name.strip().lower() in injured_qbs)

def run_predictions():
    slate = pd.read_parquet("data/pregame_features.parquet")
    out_map = get_injured_qbs_by_team()

    slate["home_qb_out"] = slate.apply(
        lambda r: is_starter_out(r["home_team"], r.get("home_qb_name", ""), out_map), axis=1
    )
    slate["away_qb_out"] = slate.apply(
        lambda r: is_starter_out(r["away_team"], r.get("away_qb_name", ""), out_map), axis=1
    )

    artifact = joblib.load("models/nfl_win_prob_v1.joblib")
    model = artifact["model"]
    feature_cols = artifact["features"]

    slate["home_win_prob"] = model.predict_proba(slate[feature_cols])[:, 1]
    slate["away_win_prob"] = 1.0 - slate["home_win_prob"]
    slate["predicted_winner"] = slate.apply(
        lambda r: r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"], axis=1
    )
    slate["confidence"] = slate[["home_win_prob", "away_win_prob"]].max(axis=1)

    display_cols = [
        "gameday", "gametime", "away_team", "home_team",
        "predicted_winner", "confidence", "home_qb_out", "away_qb_out",
    ]
    results = slate[display_cols].sort_values(["gameday", "gametime"])

    print("\n=== WEEKLY NFL PREDICTIONS (STARTER-FILTERED) ===")
    print(results.to_string(index=False, formatters={"confidence": "{:.1%}".format}))

    results.to_csv("data/latest_predictions.csv", index=False)
    print("\nSaved predictions to data/latest_predictions.csv")

if __name__ == "__main__":
    run_predictions()
