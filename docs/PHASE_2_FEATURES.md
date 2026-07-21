# Phase 2 — Feature Engineering

> Part of the **Football Expected Goals (xG) Model** project.
> Goal of this phase: turn raw shot data into meaningful features a model can learn from — and validate that each feature actually carries signal.

---

## 1. Objective

A model can't "watch" a shot — it needs numbers describing the situation. Phase 1 gave us raw shots, but the key piece of information, `location`, was just a coordinate like `[110.3, 43.2]`. This phase converts that (and other columns) into predictive features, then checks each one genuinely relates to whether a shot is scored.

**Deliverable:** a `shot_features` table in the database (9,168 rows) containing model-ready features plus the target and the StatsBomb xG benchmark.

---

## 2. The Pitch Coordinate System

StatsBomb pitches are **120 long × 80 wide**. The attacking goal sits at **x = 120**, centred at **y = 40**, with goalposts at **y = 36** and **y = 44** (an 8-unit-wide goal).

- Higher `x` → closer to goal.
- `y` near 40 → central; `y` near 0 or 80 → out wide.

This system is the basis for calculating distance and angle.

---

## 3. Features Built

### Distance to goal
Straight-line distance from the shot to the goal centre `(120, 40)`, using Pythagoras:

```python
x_dist = 120 - shots_df['x']            # gap to goal line
y_dist = (shots_df['y'] - 40).abs()     # gap from centre
shots_df['distance'] = np.sqrt(x_dist**2 + y_dist**2)
```

### Angle to goal
How wide the goal appears from the shot's position — a head-on shot sees a wide goal, a shot from the byline sees a sliver. Computed with the standard xG angle formula:

```python
goal_width = 8
angle = np.arctan2(goal_width * x_dist,
                   x_dist**2 + y_dist**2 - (goal_width/2)**2)
shots_df['angle'] = np.where(angle < 0, angle + np.pi, angle)
```

### Context flags (1 / 0)
| Feature | Meaning | Why it matters |
|---|---|---|
| `is_header` | shot was a header | headers are harder to score |
| `is_penalty` | shot was a penalty | penalties score ~75% — a different animal |
| `is_open_play` | shot came from open play | behaves differently to set pieces |

---

## 4. Validation (does each feature carry signal?)

Rather than trust the features blindly, each was checked against the goal outcome.

### Distance & angle
| Group | Avg distance | Avg angle |
|---|---|---|
| Misses (0) | 19.86 | 0.412 |
| **Goals (1)** | **12.47** | **0.694** |

Goals occur **~7 units closer** and at a **noticeably wider angle** — exactly as football sense predicts. Both features carry clear signal.

### Penalty flag
| Group | Goal rate |
|---|---|
| Non-penalty | 10.4% |
| **Penalty** | **71.1%** |

The huge gap confirms the flag works and that penalties must be distinguished from normal shots. (97 penalties in the dataset.)

---

## 5. Output

The engineered features were saved to a new **`shot_features`** table in `football_xg.db`:

```python
features_df.to_sql('shot_features', conn, if_exists='replace', index=False)
# → 9,168 rows
```

Columns kept: `distance`, `angle`, `is_header`, `is_penalty`, `is_open_play`, the target `goal`, the `shot_statsbomb_xg` benchmark, and IDs (`player`, `team`) for later player analysis.

---

## 6. Tools Used

| Tool | Role |
|---|---|
| **pandas** | Feature creation, grouping, validation |
| **NumPy** | Distance & angle math |
| **SQLite** (`sqlite3`) | Loading Phase 1 data, saving features |
| **Jupyter Notebook** | Interactive work |

---

## 7. Outcome

✅ **Phase 2 complete.** Raw coordinates and shot metadata were converted into a validated, model-ready feature set. Every feature was checked to confirm it genuinely relates to scoring before being trusted.

---

## 8. Next — Phase 3: Modelling

With features ready, Phase 3 builds the actual xG model: a logistic regression baseline first, then a stronger gradient-boosting model (XGBoost), comparing the two.
