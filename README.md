# NFL-Predictor

An automated weekly NFL prediction engine that forecasts **game winners**, **player yardage**, and **anytime touchdown probabilities** every Sunday morning before kickoff.

Instead of relying on static season averages or manual spreadsheets, the system automatically gathers fresh play-by-play data, checks real-time Sunday morning injury reports, and publishes updated predictions straight to this repository.

---

## What It Predicts Each Week

* **Game Win Probabilities:** Calculates each team's chance of winning based on dynamic team power ratings, recent offensive/defensive efficiency, rest days, home-field advantage, and whether a starting quarterback is ruled out.
* **Player Yardage & Boom Ceilings:** Projects expected **receiving yards** (WR/TE), **passing yards** (QB), and **rushing yards** (RB), alongside a **75th-percentile ceiling** showing what a player can hit when game script breaks their way.
* **Anytime Touchdown Odds:** Estimates each skill player's probability of scoring a touchdown (plus fair implied odds) using red-zone touches, goal-line carries, and Vegas team scoring expectations.

---

## How It Works (Without the Jargon)

1. **Strips Out "Garbage Time":** Only evaluates competitive plays (when win probability is between 5% and 95%) so late-game blowout stats don't skew team strength.
2. **Weights Recent Form Higher:** Uses rolling averages that give more weight to how a team or player has performed over the last few weeks rather than months ago.
3. **Checks Live Sunday Morning Injuries:** Automatically queries ESPN's live injury feed right before kickoff so inactive players are removed and starting QB injuries are factored in.
4. **Runs Hands-Free Every Sunday:** Powered by **GitHub Actions**, the pipeline wakes up automatically on Sunday mornings, updates all projections, and saves fresh CSV tables right into the `data/` folder.

---

## Where to See the Latest Predictions

You don't even need to run code to view this week's picks—the automated bot saves updated results directly in this repo:

* **Weekly Game Winners:** [`data/latest_predictions.csv`](data/latest_predictions.csv)
* **Weekly Player Yards & Touchdown Odds:** [`data/latest_player_predictions.csv`](data/latest_player_predictions.csv)

---

## Running It Locally (Optional)

If you want to run projections or look up a specific player on your own computer:

```bash
# 1. Install required packages
pip install nflreadpy pandas numpy scikit-learn joblib pyarrow requests

# 2. Generate this week's game predictions
python src/predict_sunday.py

# 3. Look up a specific player's yardage ceiling & TD probability
python src/predict_players.py --player "Justin Jefferson"
