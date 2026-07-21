Football xG-Model

Building an Expected Goals (xG) model from StatsBomb data, then using it to look at player performance and test whether xG still works when you move between leagues.
Still a work in progress — building it phase by phase.

What's xG?

xG is basically the chance a shot had of being a goal, from 0 to 1. A tap-in might be 0.9, a long shot might be 0.03. It's useful because goals are rare and a bit lucky, so xG gives a fairer picture of how good the chances actually were.
I'm building the model from scratch to learn how it works, not just using someone else's.

What I want to do:
- Build the model (start simple, then try XGBoost)
- Evaluate it properly (not just accuracy — calibration, log-loss, etc.)
- Use SHAP to see which features matter most
- Test something I'm curious about: if you train xG on one league, does it still hold up in another?
- Use it to compare players — who scores more/less than their xG says
- Turn it into a small Streamlit app at the end

Tools:
- Python
- pandas
- NumPy
- SQLite
- scikit-learn
- XGBoost
- SHAP
- Matplotlib/seaborn
- Streamlit
- Data from StatsBomb open data.

Structure:
- data/         the shots database (SQLite)
- notebooks/    one notebook per phase
- docs/         write-up for each phase

Progress:
-  Phase 0 — Setup (done)
-  Phase 1 — Get the data + store it in SQLite (notes) (done)
-  Phase 2 — Feature engineering (distance, angle, etc.)
-  Phase 3 — Build the model
-  Phase 4 — Evaluate it
-  Phase 5 — Cross-league test
-  Phase 6 — Player analysis
-  Phase 7 — Streamlit app
-  Phase 8 — Final cleanup

Data:
Using La Liga 2015/16 for now : 380 matches, 9,168 shots, about 11% of them goals.
