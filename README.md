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
*Last automatically updated: **2026-11-22 22:00 UTC***

### Current Slate: Game Picks & Vegas +EV Edges

| Matchup | Kickoff | Pick | Win Prob | Vegas Edge | Signal | Weather |
| --- | --- | --- | --- | --- | --- | --- |
| MIA @ BUF | 2026-11-22 1:00 PM ET | BUF (-1820) | 84.8% | -6.3% | Market Aligned | 65°F • Wind 11 mph |
| NO @ CHI | 2026-11-22 1:00 PM ET | CHI (-670) | 80.2% | -3.5% | Market Aligned | 55°F • Wind 10 mph • Precip 59% |
| ARI @ KC | 2026-11-22 1:00 PM ET | KC (-1460) | 79.4% | -10.5% | Market Aligned | 48°F • Wind 16 mph |
| LV @ DEN | 2026-11-22 4:25 PM ET | DEN (-650) | 75.1% | -8.2% | Market Aligned | 43°F • Wind 7 mph |
| TEN @ DAL | 2026-11-22 1:00 PM ET | DAL (-359) | 72.8% | -2.4% | Market Aligned | Dome / Indoor (70°F) |
| CIN @ WAS | 2026-11-23 8:15 PM ET | CIN (-296) | 72.5% | +0.7% |  Slight Edge | 51°F • Wind 13 mph |
| IND @ HOU | 2026-11-19 8:15 PM ET | HOU (-750) | 70.8% | -14.0% | Market Aligned | 68°F • Wind 3 mph |
| JAX @ NYG | 2026-11-22 4:25 PM ET | JAX (-261) | 70.5% | +1.0% |  Slight Edge | 59°F • Wind 7 mph |
| TB @ DET | 2026-11-22 1:00 PM ET | DET (-221) | 67.8% | +1.6% |  Slight Edge | Dome / Indoor (70°F) |
| MIN @ SF | 2026-11-22 8:20 PM ET | SF (-190) | 63.4% | +0.5% |  Slight Edge | 53°F • Wind 4 mph |
| PIT @ PHI | 2026-11-22 1:00 PM ET | PIT (-196) | 58.8% | -4.8% | Market Aligned | 63°F • Wind 8 mph |
| NYJ @ LAC | 2026-11-22 4:05 PM ET | NYJ (152) | 53.7% | +15.6% |  +EV Value | Dome / Indoor (70°F) |
| BAL @ CAR | 2026-11-22 1:00 PM ET | CAR (-141) | 52.6% | -3.6% | Market Aligned | 59°F • Wind 8 mph |

### Top 10 Anytime Touchdown Leaders

| Player | Pos | Matchup | Anytime TD % | Fair Odds |
| --- | --- | --- | --- | --- |
| Jahmyr Gibbs | RB | DET vs. TB | 78.4% | -362.0 |
| James Cook | RB | BUF vs. MIA | 72.0% | -257.0 |
| Kenneth Walker III | RB | KC vs. ARI | 67.9% | -212.0 |
| Chuba Hubbard | RB | CAR vs. BAL | 64.5% | -181.0 |
| Christian McCaffrey | RB | SF vs. MIN | 63.8% | -176.0 |
| Bucky Irving | RB | TB vs. DET | 62.8% | -169.0 |
| Ja'Marr Chase | WR | CIN vs. WAS | 60.2% | -151.0 |
| Javonte Williams | RB | DAL vs. TEN | 58.0% | -138.0 |
| CeeDee Lamb | WR | DAL vs. TEN | 56.1% | -128.0 |
| Tony Pollard | RB | TEN vs. DAL | 55.2% | -123.0 |
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
