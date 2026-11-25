# NFL-Predictor

** [Launch the Live Interactive Dashboard](https://nfl-predictor-prem.streamlit.app/)**

An automated weekly NFL prediction engine that forecasts **game winners**, **player yardage & receptions**, and **anytime touchdown probabilities** every Sunday morning before kickoff and grades its own accuracy every Tuesday after Monday Night Football.

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

<!-- LIVE_PICKS_START -->
*Last automatically updated: **2026-11-25 05:43 UTC***

### Current Slate: Game Picks & Vegas +EV Edges

| Matchup | Kickoff | Pick | Win Prob | Vegas Edge | Signal | Weather |
| --- | --- | --- | --- | --- | --- | --- |
| TEN @ JAX | 2026-11-29 1:00 PM ET | JAX (-1120) | 82.5% | -5.6% | Market Aligned | 68°F • Wind 3 mph |
| NO @ CIN | 2026-11-29 1:00 PM ET | CIN (-588) | 79.2% | -2.9% | Market Aligned | 52°F • Wind 9 mph |
| PHI @ DAL | 2026-11-26 4:30 PM ET | DAL (-471) | 73.9% | -5.4% | Market Aligned | 50°F • Wind 14 mph |
| GB @ LA | 2026-11-25 8:00 PM ET | LA (-134) | 73.4% | +18.3% |  +EV Value | Dome / Indoor (70°F) |
| NE @ LAC | 2026-11-29 4:05 PM ET | NE (-104) | 69.6% | +20.6% |  +EV Value | Dome / Indoor (70°F) |
| ATL @ MIN | 2026-11-29 1:00 PM ET | MIN (-321) | 64.2% | -9.1% | Market Aligned | Dome / Indoor (70°F) |
| BAL @ HOU | 2026-11-29 1:00 PM ET | HOU (-311) | 63.8% | -8.9% | Market Aligned | 76°F • Wind 11 mph |
| SEA @ SF | 2026-11-29 4:25 PM ET | SF (-179) | 63.7% | +2.0% |  Slight Edge | 63°F • Wind 10 mph • Precip 63% |
| CHI @ DET | 2026-11-26 1:00 PM ET | DET (102) | 61.7% | +14.2% |  +EV Value | Dome / Indoor (70°F) |
| LV @ CLE | 2026-11-29 8:20 PM ET | CLE (-219) | 59.0% | -6.9% | Market Aligned | 46°F • Wind 13 mph • Precip 58% |
| WAS @ ARI | 2026-11-29 4:25 PM ET | WAS (-337) | 58.1% | -16.0% | Market Aligned | 74°F • Wind 5 mph |
| KC @ BUF | 2026-11-26 8:20 PM ET | BUF (-149) | 56.8% | -0.7% | Market Aligned | 34°F • Wind 17 mph • Precip 96% |
| NYJ @ MIA | 2026-11-29 1:00 PM ET | NYJ (-195) | 56.5% | -7.1% | Market Aligned | 77°F • Wind 10 mph |
| DEN @ PIT | 2026-11-27 3:00 PM ET | DEN (-197) | 55.4% | -8.3% | Market Aligned | 39°F • Wind 12 mph • Precip 73% |
| CAR @ TB | 2026-11-30 8:15 PM ET | TB (-155) | 54.9% | -3.5% | Market Aligned | 70°F • Wind 4 mph |
| NYG @ IND | 2026-11-29 1:00 PM ET | NYG (-303) | 53.1% | -19.2% | Market Aligned | 45°F • Wind 5 mph • Precip 56% |

### Top 10 Anytime Touchdown Leaders

| Player | Pos | Matchup | Anytime TD % | Fair Odds |
| --- | --- | --- | --- | --- |
| Jahmyr Gibbs | RB | DET vs. CHI | 78.0% | -355.0 |
| James Cook | RB | BUF vs. KC | 63.4% | -173.0 |
| Chuba Hubbard | RB | CAR vs. TB | 62.7% | -168.0 |
| Bucky Irving | RB | TB vs. CAR | 61.7% | -161.0 |
| Kenneth Walker III | RB | KC vs. BUF | 60.0% | -150.0 |
| Javonte Williams | RB | DAL vs. PHI | 58.5% | -141.0 |
| Christian McCaffrey | RB | SF vs. SEA | 55.8% | -126.0 |
| CeeDee Lamb | WR | DAL vs. PHI | 55.0% | -122.0 |
| Bhayshul Tuten | RB | JAX vs. TEN | 53.8% | -117.0 |
| Puka Nacua | WR | LA vs. GB | 52.9% | -113.0 |
<!-- LIVE_PICKS_END -->

* **Interactive Web App :** **[nfl-predictor-prem.streamlit.app](https://nfl-predictor-prem.streamlit.app/)** — Browse weekly game picks, inspect full head-to-head offensive breakdowns, search any player, and view live season accuracy metrics.
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
python src/predict_players.py --player "Justin Jefferson"
python src/predict_players.py --team KC

# 5. Grade completed games and player props against final box scores
python src/evaluate_results.py
