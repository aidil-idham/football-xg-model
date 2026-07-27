# Phase 4 — Evaluation Deep-Dive

> Part of the **Football Expected Goals (xG) Model** project.
> Goal of this phase: go beyond single numbers and understand the model in depth — are its probabilities honest, and what drives its predictions?

---

## 1. Objective

Phase 3 gave headline metrics (AUC, log loss, Brier). But good evaluation means understanding *how* and *why* a model behaves. This phase does that with six visual diagnostics covering calibration, interpretability, ranking ability, and realism.

**Deliverable:** a full evaluation suite of six charts, each answering a specific question about the model.

---

## 2. Calibration — are the probabilities honest?

A calibration curve groups shots by predicted xG and checks the actual goal rate in each group. If the model is honest, points sit on the diagonal (predicted = actual).

**Finding:** The logistic regression baseline hugs the diagonal closely, especially in the low-xG range (0–0.4) where ~89% of shots live — so it's honest where it matters most. XGBoost showed an erratic spike (a group of confident predictions that scored 0%), visually confirming it was *worse calibrated* than the simple baseline. This reinforces the Phase 3 conclusion: the simple model was the better choice here.

*(chart: calibration_curve.png)*

---

## 3. SHAP — what drives the predictions?

SHAP explains how much each feature pushes each prediction up or down.

**Feature importance ranking:** `angle` (0.73) > `distance` (0.56) > `is_header` (0.26) > `is_open_play` (0.12) > `is_penalty` (0.02).

**Direction (all match football intuition):**
- Wide **angle** → pushes xG **up**; narrow angle → down.
- Short **distance** → pushes xG **up**; long distance → down.
- **Header** → pushes xG **down** (harder to score).
- **Penalty** → pushes xG **up** (small effect, only 97 penalties).

Every learned pattern matches real football knowledge — strong evidence the model learned genuine signal, not noise.

*(charts: shap_summary_dot.png, shap_summary_bar.png)*

---

## 4. Distance dependence — the shape of the relationship

A SHAP dependence plot shows how `distance` affects xG across its full range. xG declines steadily as distance grows, dropping off sharply beyond ~25–30 units. The colouring reveals an **interaction**: at the same distance, wider angles (red) score higher than narrow ones (blue) — so distance and angle work together, not independently.

*(chart: shap_dependence_distance.png)*

---

## 5. ROC curve — separating goals from misses

All three models bow well above the random diagonal:
- Baseline AUC **0.810**
- XGBoost AUC **0.813**
- StatsBomb AUC **0.851**

The simple model performs close to the professional benchmark using only 5 features.

*(chart: roc_curve.png)*

---

## 6. Distribution — do the xG values look realistic?

Both the baseline and StatsBomb predict many low-xG shots and few high ones — matching reality, where most shots are poor chances. The baseline's distribution closely tracks StatsBomb's, confirming the model produces realistic values rather than distorted ones.

*(chart: xg_distribution.png)*

---

## 7. Key Takeaways

1. **The model is well-calibrated** where most shots occur — its probabilities can be trusted.
2. **The simple baseline is better calibrated than XGBoost**, visually reinforcing that model complexity didn't help on this feature set.
3. **The model learned football-sensible patterns** (SHAP), giving confidence it captured real signal.
4. **Distance and angle interact** — a subtle, real insight surfaced by the dependence plot.
5. **Predicted xG values are realistic**, closely matching the professional benchmark's distribution.

---

## 8. Tools Used

| Tool | Role |
|---|---|
| **scikit-learn** | Calibration curve, ROC curve, metrics |
| **SHAP** | Feature importance & interpretability |
| **Matplotlib** | All charts |
| **XGBoost / LogisticRegression** | The models being evaluated |

---

## 9. Outcome

✅ **Phase 4 complete.** The model was evaluated thoroughly — not just scored, but understood. Its probabilities are honest, its logic is football-sensible, and its outputs are realistic. This depth of evaluation is what distinguishes a rigorous model from one that merely produces plausible numbers.

---

## 10. Next — Phase 5: Cross-League Test

The signature investigation: if xG is trained on one league, does it still hold up in another? (The "does xG travel?" question — inspired by players who dominate one league but struggle in a tougher one.)
