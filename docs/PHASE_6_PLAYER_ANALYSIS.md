# Phase 6 — Player Analysis: Who Beats Their xG?

> Part of the **Football Expected Goals (xG) Model** project.
> Goal of this phase: use the model to produce a real football insight — which players score more (or fewer) goals than the quality of their chances suggests.

---

## 1. Objective

An xG model isn't just an academic exercise — its real value is answering practical questions. This phase uses the model to measure **finishing ability**:

> Given the chances a player got, did they score more or fewer goals than expected?

- **Actual goals > total xG** → over-performer (clinical finisher, or lucky).
- **Actual goals < total xG** → under-performer (wasteful, or unlucky).

This is exactly the kind of analysis clubs use to separate genuinely good finishers from players flattered by the chances they receive.

---

## 2. Method

1. Trained the model on **all** shots (not a train/test split) — because here the model is being *used* to estimate xG for every shot, not evaluated (evaluation was done in Phase 4).
2. Predicted an xG value for each of the 9,168 shots.
3. Grouped shots by player and summed:
   - `shots` — number of shots taken
   - `goals` — actual goals scored
   - `xg` — total expected goals (sum of shot xG)
   - `goals_minus_xg` — the key over/under-performance figure
4. Kept only players with **20+ shots** to avoid unreliable small samples.

151 players qualified.

---

## 3. Results

### Top over-performers (clinical finishers)

| Player | Shots | Goals | xG | Goals − xG |
|---|---|---|---|---|
| Luis Suárez | 139 | 40 | 25.79 | **+14.21** |
| Gareth Bale | 81 | 19 | 8.03 | +10.97 |
| Antoine Griezmann | 92 | 22 | 13.55 | +8.45 |
| Cristiano Ronaldo | 228 | 35 | 28.50 | +6.50 |
| Karim Benzema | 98 | 24 | 18.23 | +5.77 |

### Top under-performers

| Player | Shots | Goals | xG | Goals − xG |
|---|---|---|---|---|
| Álvaro Vázquez | 48 | 5 | 8.49 | −3.49 |
| Diego Godín | 26 | 1 | 4.18 | −3.18 |
| Carlos Vela | 61 | 5 | 7.78 | −2.78 |

---

## 4. Key Finding — the model validates itself

The biggest over-performers are **exactly** the players you'd expect: Suárez, Bale, Griezmann, Ronaldo, Benzema — some of the best finishers in world football in 2015/16. This is strong validation:

> World-class finishers *should* consistently beat their xG, and the analysis independently identified them as the top over-performers.

Suárez's season stands out — **40 goals from 25.8 xG** (he won the Pichichi as top scorer that year). The model correctly captured that he was converting chances at an elite rate.

A scatter of xG vs actual goals per player makes this instantly readable: elite finishers sit clearly above the "average finishing" diagonal, while most players cluster along it.

*(chart: player_xg_vs_goals.png)*

---

## 5. Why This Matters

This phase turns the model into a **decision tool**. The same logic clubs use in recruitment — is a striker's goal record backed by chance quality, or are they over/under-performing their xG? — is reproduced here. It's the "so what" that connects a technical model to real footballing value.

---

## 6. Honest Notes

- Over/under-performance can reflect **finishing skill OR luck** — a single season can't fully separate the two. Repeated seasons above xG is stronger evidence of genuine skill.
- The 20-shot minimum reduces noise but doesn't eliminate it; the smallest-sample players are still less reliable.

---

## 7. Tools Used

| Tool | Role |
|---|---|
| **pandas** | Grouping shots into player summaries |
| **scikit-learn** | The xG model used to score shots |
| **Matplotlib** | The xG-vs-goals scatter chart |

---

## 8. Outcome

✅ **Phase 6 complete.** The model was used to produce a genuine football insight — identifying clinical finishers and under-performers — and the results matched real-world expectations, validating both the model and the analysis.

---

## 9. Next — Phase 7: Streamlit App

Wrapping the model into an interactive app so anyone can input a shot's details and get an xG value, or explore player performance — turning the project from notebooks into a usable tool.
