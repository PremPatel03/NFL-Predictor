"""
Automatically injects the current week's Game Picks (+EV & Weather) and
Top 10 Anytime Touchdown Leaders into `README.md` between:
  <!-- LIVE_PICKS_START -->
  <!-- LIVE_PICKS_END -->
"""

from datetime import datetime, timezone
import os
import re
import pandas as pd

README_PATH = "README.md"
GAMES_CSV = "data/latest_predictions.csv"
PLAYERS_CSV = "data/latest_player_predictions.csv"

def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Converts a DataFrame into a clean GitHub-flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(v) if pd.notna(v) else "—" for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def build_live_markdown() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [f"*Last automatically updated: **{timestamp}***\n"]

    if os.path.exists(GAMES_CSV):
        games = pd.read_csv(GAMES_CSV)
        if not games.empty:
            view = pd.DataFrame()
            view["Matchup"] = games["away_team"] + " @ " + games["home_team"]
            gametime_col = games["gametime"].fillna("").astype(str) if "gametime" in games.columns else pd.Series("", index=games.index)
            ml_col = games["pick_ml_str"].fillna("N/A").astype(str) if "pick_ml_str" in games.columns else pd.Series("N/A", index=games.index)

            view["Kickoff"] = (games["gameday"].astype(str) + " " + gametime_col).str.strip()
            view["Pick"] = games["predicted_winner"].astype(str) + " (" + ml_col + ")"
            view["Win Prob"] = games["win_confidence"].map(lambda x: f"{x:.1f}%")
            if "model_edge" in games.columns:
                view["Vegas Edge"] = games["model_edge"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
            if "ev_flag" in games.columns:
                view["Signal"] = games["ev_flag"]
            if "weather_summary" in games.columns:
                view["Weather"] = games["weather_summary"]

            sections.append("### Current Slate: Game Picks & Vegas +EV Edges\n")
            sections.append(_df_to_markdown_table(view))
            sections.append("")

    if os.path.exists(PLAYERS_CSV):
        players = pd.read_csv(PLAYERS_CSV)
        if not players.empty and "td_prob" in players.columns:
            top_td = (
                players.dropna(subset=["td_prob"])
                .drop_duplicates(subset=["player_display_name", "team"])
                .sort_values("td_prob", ascending=False)
                .head(10)
            )
            if not top_td.empty:
                td_view = pd.DataFrame()
                td_view["Player"] = top_td["player_display_name"]
                td_view["Pos"] = top_td["position"]
                td_view["Matchup"] = top_td["team"] + " vs. " + top_td["opponent"]
                td_view["Anytime TD %"] = top_td["td_prob"].map(lambda x: f"{x:.1f}%")
                td_view["Fair Odds"] = top_td["fair_td_odds"]

                sections.append("### Top 10 Anytime Touchdown Leaders\n")
                sections.append(_df_to_markdown_table(td_view))
                sections.append("")

    return "\n".join(sections).strip()

def update_readme() -> None:
    if not os.path.exists(README_PATH):
        print("README.md not found.")
        return

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"<!-- LIVE_PICKS_START -->.*?<!-- LIVE_PICKS_END -->", re.DOTALL)
    if not pattern.search(content):
        print("Markers <!-- LIVE_PICKS_START --> and <!-- LIVE_PICKS_END --> not found in README.md.")
        return

    live_block = f"<!-- LIVE_PICKS_START -->\n{build_live_markdown()}\n<!-- LIVE_PICKS_END -->"
    updated = pattern.sub(live_block, content)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)
    print("Successfully updated README.md with live weekly picks.")

if __name__ == "__main__":
    update_readme()
