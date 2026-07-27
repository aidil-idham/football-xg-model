# Phase 5 — Cross-League Test: Does xG Travel?

> Part of the **Football Expected Goals (xG) Model** project.
> **This is the project's signature investigation.**
> Goal: test whether an xG model trained on one league still works on a different league — or whether xG "breaks" when it travels.

---

## 1. The Question

A common observation in football: players who dominate one league can struggle in another (e.g. a striker who scores freely in a weaker division but stalls in a tougher one). This raises a real analytics question:

> **If an xG model is trained on one league, does it still make good predictions in another?**

A standard xG model judges a shot only by its own circumstances (distance, angle, etc.) — it can't see league quality, defensive intensity, or opposition strength. So in theory, a model trained on one league might not fully transfer to another. This phase tests that directly.

---

## 2. Method

- **Train league:** La Liga 2015/16 (9,168 shots) — the model from earlier phases.
- **Test league:** Bundesliga 2015/16 (840 shots) — same season, different country/style.
- The La Liga-trained model was applied to Bundesliga shots **without any retraining**.
- Bundesliga shots went through the **identical feature engineering** (distance, angle, header/penalty/open-play flags) so the comparison is fair.

Choosing the same season controls for era, isolating the effect of the *league* itself. La Liga (possession-based) and Bundesliga (more open, transitional) differ in style, making this a meaningful test.

---

## 3. Results

Overall goal rates were similar (La Liga 11.1%, Bundesliga 10.7%), but model performance dropped when crossing leagues:

| Metric | On La Liga (home) | On Bundesliga (away) |
|---|---|---|
| ROC AUC | 0.810 | **0.772** |
| Log loss | 0.2744 | 0.2835 |
| Brier score | 0.0788 | 0.0807 |

A calibration comparison showed the model stays fairly honest on Bundesliga in the low-xG range (where most shots are), but drifts from the diagonal more than on La Liga, particularly at higher xG values.

---

## 4. Finding

**xG partially travels — but not perfectly.**

- The model still performs clearly better than random on Bundesliga (0.77 AUC), so the **universal fundamentals of a good chance** (distance, angle) do carry across leagues.
- But performance **measurably degrades** (~0.04 AUC drop, worse log loss, looser calibration), confirming that **league-specific factors the model can't see cause real transfer loss.**

This is the "Delap effect" quantified: the same underlying quality doesn't map identically across different competitive environments.

---

## 5. Honest Limitations

- **Small test sample:** StatsBomb's open data for Bundesliga 2015/16 contained only 840 shots (a partial season). This makes the high-xG calibration bins noisy — the sharp swings at high predicted xG are likely sample artefacts, not real model failure. The low-to-mid xG range (most shots) is the more reliable part of the comparison.
- Only one train/test league pair was tested; a fuller study would repeat this across several league pairs to confirm the pattern generalises.

---

## 6. Why This Matters

This phase turns the project from "another xG model" into an **investigation**:
- A hypothesis was formed (xG won't fully transfer across leagues).
- An experiment was designed to test it (train on one league, test on another, no retraining).
- The result was measured (a specific AUC drop) and interpreted honestly (with sample-size caveats).

That hypothesis → experiment → result → honest interpretation loop is the core of real data-science work.

---

## 7. Tools Used

| Tool | Role |
|---|---|
| **statsbombpy** | Pulling the second league's data |
| **pandas / NumPy** | Feature engineering on the new league |
| **scikit-learn** | Applying the model, metrics, calibration |
| **Matplotlib** | Cross-league calibration chart |

---

## 8. Outcome

✅ **Phase 5 complete.** The signature question was answered with evidence: xG partially transfers across leagues, capturing universal shot fundamentals but degrading measurably due to unobserved league differences — a finding backed by ranking metrics and calibration, and reported with honest attention to sample-size limits.

---

## 9. Next — Phase 6: Player Analysis

Using the model to compare players: actual goals vs expected goals, to find who over- and under-performs their xG.
