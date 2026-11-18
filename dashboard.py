"""
NFL-Predictor Interactive Web Dashboard (`Dashboard.py`).

Reads weekly pre-kickoff predictions (`data/latest_predictions.csv`,
`data/latest_player_predictions.csv`) and cumulative grading logs
(`data/season_game_tracker.csv`, `data/season_player_tracker.csv`)
to provide a live, interactive UI for game picks, matchup breakdowns,
player prop lookups, and model calibration tracking.
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NFL-Predictor | Weekly Win & Player Prop",
    page_icon="🏈",
    layout="wide",
)

GAME_PRED_PATH = "data/latest_predictions.csv"
PLAYER_PRED_PATH = "data/latest_player_predictions.csv"
GAME_TRACKER_PATH = "data/season_game_tracker.csv"
PLAYER_TRACKER_PATH = "data/season_player_tracker.csv"

@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    """Safely loads a CSV file if it exists, returning an empty DataFrame otherwise."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def format_matchup_label(row: pd.Series) -> str:
    """Formats a readable game label for dropdown selectors."""
    slot = f" ({row['time_slot']})" if "time_slot" in row and pd.notna(row["time_slot"]) else ""
    return f"{row['away_team']} @ {row['home_team']}{slot}"

# ---------------------------------------------------------------------------
# Header & Top KPI Summary Bar
# ---------------------------------------------------------------------------
st.title("🏈 NFL-Predictor Dashboard")
st.caption(
    "Automated weekly game win probabilities, player yardage ceilings, and Anytime TD odds — "
    "updated every Sunday morning before kickoff."
)

games_df = load_csv(GAME_PRED_PATH)
players_df = load_csv(PLAYER_PRED_PATH)
game_tracker_df = load_csv(GAME_TRACKER_PATH)
player_tracker_df = load_csv(PLAYER_TRACKER_PATH)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Games on Current Slate", len(games_df) if not games_df.empty else "—")

with kpi2:
    if not games_df.empty and "win_confidence" in games_df.columns:
        top_pick = games_df.sort_values("win_confidence", ascending=False).iloc[0]
        st.metric(
            "Highest-Confidence Pick",
            f"{top_pick['predicted_winner']}",
            delta=f"{top_pick['win_confidence']:.1f}% Win Prob",
        )
    else:
        st.metric("Highest-Confidence Pick", "—")

with kpi3:
    if not players_df.empty and "td_prob" in players_df.columns:
        td_only = players_df.dropna(subset=["td_prob"]).sort_values("td_prob", ascending=False)
        if not td_only.empty:
            top_td = td_only.iloc[0]
            st.metric(
                "Top Anytime TD Odds",
                f"{top_td['player_display_name']}",
                delta=f"{top_td['td_prob']:.1f}% Anytime TD",
            )
        else:
            st.metric("Top Anytime TD Odds", "—")
    else:
        st.metric("Top Anytime TD Odds", "—")

with kpi4:
    if not game_tracker_df.empty and "correct" in game_tracker_df.columns:
        wins = int(game_tracker_df["correct"].sum())
        total = len(game_tracker_df)
        st.metric("Season Pick Record", f"{wins}-{total - wins} ({wins / total:.1%})")
    else:
        st.metric("Season Pick Record", "Tracking Live")

st.divider()

# Main Navigation Tabs
tab_games, tab_matchups, tab_players, tab_tracker = st.tabs(
    [
        "Weekly Game Picks",
        "Game Matchup Breakdown",
        "Player Prop Explorer",
        "Season Accuracy Tracker",
    ]
)

# TAB 1: WEEKLY GAME PICKS
with tab_games:
    if games_df.empty:
        st.warning("No game predictions found yet. Run `python src/build_features.py && python src/predict_sunday.py`.")
    else:
        col_filter1, col_filter2 = st.columns([2, 1])
        with col_filter1:
            if "time_slot" in games_df.columns:
                slots = ["All Kickoff Slots"] + sorted(games_df["time_slot"].dropna().unique().tolist())
                selected_slot = st.selectbox("Filter by Kickoff Window", slots)
            else:
                selected_slot = "All Kickoff Slots"
        with col_filter2:
            min_conf = st.slider("Minimum Win Confidence (%)", 50, 90, 50)

        filtered_games = games_df.copy()
        if selected_slot != "All Kickoff Slots" and "time_slot" in filtered_games.columns:
            filtered_games = filtered_games[filtered_games["time_slot"] == selected_slot]
        if "win_confidence" in filtered_games.columns:
            filtered_games = filtered_games[filtered_games["win_confidence"] >= min_conf]

        st.subheader(f"Projected Game Outcomes ({len(filtered_games)} Matchups)")

        for _, row in filtered_games.iterrows():
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2.2, 2.2, 2.5, 1.5])
                with c1:
                    st.markdown(f"**{row['away_team']} @ {row['home_team']}**")
                    kickoff_str = f"{row['gameday']}"
                    if "gametime" in row and pd.notna(row["gametime"]):
                        kickoff_str += f" • {row['gametime']}"
                    if "time_slot" in row and pd.notna(row["time_slot"]):
                        kickoff_str += f" ({row['time_slot']})"
                    st.caption(kickoff_str)

                with c2:
                    st.markdown(f"Projected Winner: **{row['predicted_winner']}**")
                    st.caption(f"Model Confidence: **{row['win_confidence']:.1f}%**")

                with c3:
                    home_p = float(row["home_win_prob"]) / 100.0
                    st.progress(
                        home_p,
                        text=f"{row['away_team']} {row['away_win_prob']:.1f}%  |  {row['home_team']} {row['home_win_prob']:.1f}%",
                    )

                with c4:
                    conf = float(row["win_confidence"])
                    if conf >= 75.0:
                        st.success("Lock Pick")
                    elif conf >= 60.0:
                        st.info("Solid Lean")
                    else:
                        st.warning("Toss-Up")

# TAB 2: FULL GAME OFFENSIVE BREAKDOWN
with tab_matchups:
    if games_df.empty or players_df.empty:
        st.info("Generate both game and player predictions first to inspect full offensive matchups.")
    else:
        matchup_options = {format_matchup_label(r): (r["away_team"], r["home_team"]) for _, r in games_df.iterrows()}
        chosen_label = st.selectbox("Select a Matchup", list(matchup_options.keys()))
        away_tm, home_tm = matchup_options[chosen_label]

        team_col1, team_col2 = st.columns(2)
        for col_box, tm, role in [(team_col1, away_tm, "Away Offense"), (team_col2, home_tm, "Home Offense")]:
            with col_box:
                st.subheader(f"{tm} ({role})")
                tm_players = players_df[players_df["team"] == tm].copy()
                if tm_players.empty:
                    st.write("No active player projections available.")
                else:
                    tm_players = tm_players.sort_values(["stat_type", "proj_yards"], ascending=[True, False])
                    display_cols = [
                        c for c in ["player_display_name", "position", "stat_type", "proj_yards", "ceiling_75", "td_prob", "fair_td_odds"]
                        if c in tm_players.columns
                    ]
                    st.dataframe(
                        tm_players[display_cols].rename(
                            columns={
                                "player_display_name": "Player",
                                "position": "Pos",
                                "stat_type": "Category",
                                "proj_yards": "Projected",
                                "ceiling_75": "75th Ceiling",
                                "td_prob": "Anytime TD %",
                                "fair_td_odds": "Fair TD Odds",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

# TAB 3: PLAYER PROP & ANYTIME TD EXPLORER
with tab_players:
    if players_df.empty:
        st.warning("No player projections found yet. Run `python src/predict_players.py`.")
    else:
        f1, f2, f3 = st.columns([2.5, 1.5, 1.5])
        with f1:
            search_name = st.text_input("Search Player Name", placeholder="e.g., Zay Flowers, Jahmyr Gibbs, Patrick Mahomes")
        with f2:
            stat_choices = ["All Categories"] + sorted(players_df["stat_type"].dropna().unique().tolist())
            selected_stat = st.selectbox("Stat Category", stat_choices)
        with f3:
            pos_choices = ["All Positions"] + sorted(players_df["position"].dropna().unique().tolist())
            selected_pos = st.selectbox("Position", pos_choices)

        filtered_players = players_df.copy()
        if search_name.strip():
            filtered_players = filtered_players[
                filtered_players["player_display_name"].str.contains(search_name.strip(), case=False, na=False)
            ]
        if selected_stat != "All Categories":
            filtered_players = filtered_players[filtered_players["stat_type"] == selected_stat]
        if selected_pos != "All Positions":
            filtered_players = filtered_players[filtered_players["position"] == selected_pos]

        sort_by = st.radio(
            "Sort Leaderboard By",
            ["Projected Yards / Receptions", "Anytime TD Probability"],
            horizontal=True,
        )
        if sort_by == "Anytime TD Probability" and "td_prob" in filtered_players.columns:
            filtered_players = filtered_players.sort_values("td_prob", ascending=False, na_position="last")
        else:
            filtered_players = filtered_players.sort_values("proj_yards", ascending=False)

        rename_map = {
            "player_display_name": "Player",
            "position": "Pos",
            "team": "Team",
            "opponent": "Opp",
            "stat_type": "Category",
            "proj_yards": "Projected",
            "ceiling_75": "75th-Pct Ceiling",
            "td_prob": "Anytime TD %",
            "fair_td_odds": "Fair TD Odds",
        }
        show_cols = [c for c in rename_map if c in filtered_players.columns]
        st.dataframe(
            filtered_players[show_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

# CUMULATIVE SEASON ACCURACY TRACKER
with tab_tracker:
    st.subheader("Post-Game Graded Accuracy & Calibration")
    st.caption("Automatically updated every Tuesday morning after Monday Night Football via GitHub Actions.")

    if game_tracker_df.empty and player_tracker_df.empty:
        st.info(
            "No completed weeks have been graded yet. Once games finish and `src/evaluate_results.py` runs "
            "(automatically every Tuesday at 9:00 AM EST), your season record, Brier score, and player prop MAE will appear here."
        )
    else:
        if not game_tracker_df.empty:
            st.markdown("#### Game Winner Accuracy by Week")
            weekly_summary = (
                game_tracker_df.groupby(["season", "week"], as_index=False)
                .agg(
                    Games=("correct", "count"),
                    Correct=("correct", "sum"),
                    Accuracy=("correct", "mean"),
                    Brier_Score=("brier", "mean"),
                )
            )
            weekly_summary["Accuracy"] = (weekly_summary["Accuracy"] * 100).round(1)
            weekly_summary["Brier_Score"] = weekly_summary["Brier_Score"].round(4)
            st.dataframe(weekly_summary, use_container_width=True, hide_index=True)

        if not player_tracker_df.empty:
            st.markdown("#### Player Prop Error & Ceiling Calibration")
            prop_summary = (
                player_tracker_df.groupby("stat_type", as_index=False)
                .agg(
                    Graded_Props=("abs_error", "count"),
                    Mean_Abs_Error=("abs_error", "mean"),
                    Ceiling_75_Hit_Rate=("hit_ceiling_75", "mean"),
                )
            )
            prop_summary["Mean_Abs_Error"] = prop_summary["Mean_Abs_Error"].round(1)
            prop_summary["Ceiling_75_Hit_Rate"] = (prop_summary["Ceiling_75_Hit_Rate"] * 100).round(1)
            st.dataframe(prop_summary, use_container_width=True, hide_index=True)
