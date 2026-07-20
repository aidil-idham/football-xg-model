# Phase 1 — Data Acquisition & Storage

> Part of the **Football Expected Goals (xG) Model** project.
> Goal of this phase: get a clean, well-understood pile of shots into a queryable database — the fuel the model will learn from.

---

## 1. Objective

Before any modelling, the project needs a large set of **historical shots** where two things are known for each shot:

1. **The circumstances** of the shot (where it was taken from, body part, play type, etc.)
2. **What actually happened** (goal or no goal)

This phase covers sourcing that data, understanding it, and storing it so it never has to be re-downloaded.

**Deliverable:** a SQLite database (`football_xg.db`) containing 9,168 shots from a full La Liga season, plus a documented understanding of the data's shape.

---

## 2. Data Source

| Item | Detail |
|---|---|
| **Provider** | [StatsBomb Open Data](https://github.com/statsbomb/open-data) (free tier) |
| **Access method** | `statsbombpy` Python package |
| **Competition** | La Liga (`competition_id = 11`) |
| **Season** | 2015/2016 (`season_id = 27`) |
| **Matches** | 380 (full season — 20 teams, home & away) |
| **Shots extracted** | 9,168 |

**Why StatsBomb open data?** It is free, shot-level detailed, and includes StatsBomb's *own* xG value per shot — which serves as a benchmark to validate the model against later.

**Why La Liga 2015/16?** It is the richest league in StatsBomb's open data (full Messi-era Barcelona seasons available), giving the largest volume of shots, and it is a widely recognised dataset — good for a portfolio piece.

**Why one season first?** Shots are fetched match-by-match over the network. Building the full pipeline on one complete season (≈9k shots — plenty for a first model) keeps iteration fast; scaling to more seasons is a later step once the pipeline is proven.

---

## 3. Methodology

The StatsBomb data is hierarchical — like drawers within drawers:

```
Competitions  →  Matches  →  Events  →  (filter to) Shots
```

### Step 1 — Connect and list competitions
```python
from statsbombpy import sb
competitions = sb.competitions()
```
Confirmed the data pipe works. (A `NoAuthWarning: open data access only` is expected on the free tier and is not an error.)

### Step 2 — Pull the season's matches
```python
matches = sb.matches(competition_id=11, season_id=27)
# → 380 matches
```

### Step 3 — Extract shots from every match
Looped through all 380 matches, pulled each match's events, and kept only the `Shot` events, stacking them into one table.
```python
all_shots = []
for mid in matches['match_id'].tolist():
    events = sb.events(match_id=mid)
    shots = events[events['type'] == 'Shot']
    all_shots.append(shots)
shots_df = pd.concat(all_shots, ignore_index=True)
# → 9,168 shots
```

### Step 4 — Define the target variable
Created the column the model will learn to predict: `goal` = 1 if the shot was a goal, else 0.
```python
shots_df['goal'] = (shots_df['shot_outcome'] == 'Goal').astype(int)
```

---

## 4. Key Findings (Data Understanding)

### Shot outcomes
| Outcome | Count |
|---|---|
| Off Target | 2,981 |
| Saved | 2,216 |
| Blocked | 2,081 |
| **Goal** | **1,014** |
| Wayward | 599 |
| Post | 209 |
| Saved Off Target | 35 |
| Saved to Post | 33 |

### Goal rate — **11.1%**
Only 1,014 of 9,168 shots were goals.

**Why this matters:** the data is **imbalanced** (≈89% not-goal, ≈11% goal). This has a direct consequence for how the model is evaluated:

> A naive model that predicts "no goal" for *every* shot would be **89% accurate** — yet completely useless. This is why the project evaluates with **calibration, log-loss, and Brier score** rather than accuracy. *(See Phase 4.)*

This 11% figure is also realistic for real-world football (roughly 1 in 9 shots score), confirming the data is sound.

### Benchmark sanity check
| Metric | Value |
|---|---|
| Actual goal rate | 11.1% |
| Average StatsBomb xG | 10.7% |

The two match closely — the sign of a **well-calibrated** benchmark model. StatsBomb's per-shot xG (`shot_statsbomb_xg`) will be used as a reference to validate the model built in later phases.

---

## 5. Data Storage (SQL)

The cleaned shots were stored in a **SQLite** database so the 380 downloads never need repeating, and so the data is queryable with real SQL.

```python
import sqlite3
conn = sqlite3.connect('football_xg.db')

cols_to_save = ['id', 'match_id', 'player', 'team', 'shot_statsbomb_xg',
                'shot_body_part', 'shot_technique', 'shot_type',
                'shot_outcome', 'location', 'goal']
shots_to_save = shots_df[cols_to_save].copy()
shots_to_save['location'] = shots_to_save['location'].astype(str)  # list → text for SQLite
shots_to_save.to_sql('shots', conn, if_exists='replace', index=False)
```

Verified with a SQL query:
```sql
SELECT COUNT(*) AS total_shots, SUM(goal) AS total_goals FROM shots;
-- → 9168 shots, 1014 goals
```

**Design decision — why SQLite over PostgreSQL?**
SQLite is the correctly-sized tool for a single-user analytics project of this scale. It is genuine SQL (same query language), requires zero server setup, and stores the database as a single portable file. PostgreSQL's advantages (concurrency, scale, server features) add no value at ~9k rows and one user. *Postgres is planned for a later, multi-user project to demonstrate progression.*

---

## 6. Tools Used

| Tool | Role in this phase |
|---|---|
| **Python** | Core language |
| **statsbombpy** | Fetching StatsBomb open data |
| **pandas** | Data wrangling, filtering, table assembly |
| **SQLite** (`sqlite3`) | Persistent, queryable storage |
| **SQL** | Verifying stored data |
| **Jupyter Notebook** | Interactive exploration |

---

## 7. Outcome

✅ **Phase 1 complete.** A pipeline was built that:
- connects to StatsBomb open data,
- pulls a full La Liga season (380 matches),
- extracts and cleans 9,168 shots,
- defines the prediction target (`goal`),
- validates data quality against a known benchmark, and
- stores everything in a queryable SQLite database.

---

## 8. Next — Phase 2: Feature Engineering

The raw `location` coordinate (e.g. `[110.3, 43.2]`) is not yet usable by a model. Phase 2 converts it into meaningful predictive features — most importantly **distance to goal** and **angle to goal** — the inputs the xG model actually learns from.
