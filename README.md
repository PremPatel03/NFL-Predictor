# 🏈 NFL-Predictor

** [Launch the Live Interactive Dashboard](https://nfl-predictor-prem.streamlit.app/)**

An automated weekly NFL prediction engine that forecasts **game winners**, **player yardage & receptions**, and **anytime touchdown probabilities** every Sunday morning before kickoff—and grades its own accuracy every Tuesday after Monday Night Football.

Instead of relying on static season averages or manual spreadsheets, the system automatically gathers fresh play-by-play data, checks real-time Sunday morning injury reports, and publishes updated predictions straight to this repository and the live web app.

---

## What It Predicts Each Week

* **Game Win Probabilities:** Calculates each team's chance of winning across every kickoff window (`Thu Night`, `Sun 1:00 PM`, `Sun 4:00 PM`, `Sun Night`, `Mon Night`) based on dynamic team power ratings, recent offensive/defensive efficiency, rest days, home-field advantage, and starting QB injury status.
* **Player Yardage, Receptions & Boom Ceilings:** Projects expected **receiving yards** (WR/TE/RB), **receptions** (RB), **passing yards** (QB), and **rushing yards** (RB), alongside a **75th-percentile ceiling** showing what a player can hit when game script breaks their way.
* **Anytime Touchdown Odds:** Estimates each skill player's probability of scoring a touchdown (plus fair implied American odds) using red-zone touches, goal-line carries, and Vegas team scoring expectations.

---

## How It Works (Without the Jargon)

1. **Strips Out "Garbage Time":** Only evaluates competitive plays (when win probability is between 5% and 95%) so late-game blowout stats don't skew team strength.
2. **Weights Recent Form Higher:** Uses exponentially weighted rolling averages that prioritize how a team or player has performed over the last few weeks rather than months ago.
3. **Checks Live Sunday Morning Injuries:** Queries ESPN's live injury feed right before kickoff so inactive players are removed and starting QB injuries are factored in.
4. **Runs Hands-Free Every Sunday & Tuesday:** Powered by **GitHub Actions**, the pipeline generates fresh predictions every Sunday morning (`11:40 AM EST`), archives weekly snapshots, and grades completed box scores every Tuesday morning (`9:00 AM EST`) to track cumulative season accuracy.

---

## Where to See the Latest Predictions

* **Interactive Web App (Recommended):** **[nfl-predictor-prem.streamlit.app](https://nfl-predictor-prem.streamlit.app/)** — Browse weekly game picks, inspect full head-to-head offensive breakdowns, search any player, and view live season accuracy metrics.
* **Raw Weekly CSVs in This Repo:**
  * **Weekly Game Winners:** [`data/latest_predictions.csv`](data/latest_predictions.csv)
  * **Weekly Player Yards, Receptions & TD Odds:** [`data/latest_player_predictions.csv`](data/latest_player_predictions.csv)
  * **Season Game Accuracy Log:** [`data/season_game_tracker.csv`](data/season_game_tracker.csv)
  * **Season Player Prop Accuracy Log:** [`data/season_player_tracker.csv`](data/season_player_tracker.csv)

---

## Running It Locally (Optional)

If you want to run the dashboard, generate projections, or inspect a matchup on your own computer:

```bash
# 1. Install required packages
pip install -r requirements.txt
pip install nflreadpy scikit-learn joblib pyarrow requests

# 2. Launch the interactive web dashboard locally
streamlit run dashboard.py

# 3. Generate this week's game predictions
PYTHONPATH=src python src/build_features.py && PYTHONPATH=src python src/predict_sunday.py

# 4. Look up a specific player or an entire game matchup
python src/predict_players.py --player "Zay Flowers"
python src/predict_players.py --team KC

# 5. Grade completed games and player props against final box scores
python src/evaluate_results.py
