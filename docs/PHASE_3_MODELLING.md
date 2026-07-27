# Phase 3 — Modelling

> Part of the **Football Expected Goals (xG) Model** project.
> Goal of this phase: build the actual xG model — a simple baseline first, then a stronger model — and compare both honestly against the professional StatsBomb model.

---

## 1. Objective

With validated features ready from Phase 2, this phase builds the model that predicts each shot's probability of becoming a goal. The approach follows good practice: build a simple baseline, then a more powerful model, and compare — rather than jumping straight to a complex model and assuming it's best.

**Deliverable:** two trained xG models (logistic regression + XGBoost), evaluated on unseen data and benchmarked against StatsBomb's own xG.

---

## 2. Train/Test Split

A model must never be tested on the data it learned from — that would be like giving a student the exam answers in advance. The 9,168 shots were split into two piles:

```python
X = df[['distance', 'angle', 'is_header', 'is_penalty', 'is_open_play']]  # features
y = df['goal']                                                            # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **Training set:** 7,334 shots (the model learns from these)
- **Test set:** 1,834 shots (hidden away, used only to judge the model)
- `stratify=y` keeps the ~11% goal rate identical in both piles — important with imbalanced data.

---

## 3. Models Built

### Baseline — Logistic Regression
A simple, classic model that outputs a probability. It's the honest benchmark: if a fancier model can't beat this, the complexity isn't worth it.

```python
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)
```

### Stronger — XGBoost
A gradient-boosting model that builds many small decision trees, each correcting the last. Can capture more complex patterns than a straight-line model.

```python
xgb_model = XGBClassifier(n_estimators=200, max_depth=4,
                          learning_rate=0.05, eval_metric='logloss',
                          random_state=42)
xgb_model.fit(X_train, y_train)
```

---

## 4. Evaluation Metrics

Because the data is imbalanced (~11% goals), **accuracy is misleading** — a model predicting "no goal" every time would be ~89% accurate but useless. Instead, probability-appropriate metrics were used:

- **Log loss** — punishes confident-but-wrong predictions (lower is better)
- **Brier score** — average squared error of the probabilities (lower is better)
- **ROC AUC** — how well the model ranks good chances above bad ones (higher is better; 0.5 = random)

---

## 5. Results

| Metric | Baseline (LogReg) | XGBoost | StatsBomb |
|---|---|---|---|
| Log loss | 0.2744 | 0.2769 | **0.2468** |
| Brier score | 0.0788 | 0.0808 | **0.0703** |
| ROC AUC | 0.8097 | 0.8133 | **0.8506** |

---

## 6. Key Findings

**1. A simple model already performs near-professional level.**
With only 5 features (distance, angle, header, penalty, open-play), the baseline reached **0.81 AUC** — within **0.04** of StatsBomb's professional model (0.85), which uses far richer data (defender/keeper positions, detailed technique, etc.). This shows that a few well-chosen features capture most of the predictive signal.

**2. XGBoost did NOT meaningfully beat the baseline.**
The more complex model gained almost nothing (and was slightly worse on log loss/Brier). The lesson: **performance here is limited by feature quality, not model complexity.** With only 5 features, both models are near the ceiling of what that information allows. The path to closing the gap with StatsBomb is *better features*, not a fancier algorithm — a core applied-ML insight.

---

## 7. Tools Used

| Tool | Role |
|---|---|
| **scikit-learn** | Train/test split, logistic regression, metrics |
| **XGBoost** | Gradient-boosting model |
| **pandas / SQLite** | Loading the feature table |
| **Jupyter Notebook** | Interactive modelling |

---

## 8. Outcome

✅ **Phase 3 complete.** Two xG models were built, evaluated honestly on unseen data, and benchmarked against a professional model. Just as importantly, the *why* behind the results is understood: features, not model complexity, are the bottleneck.

---

## 9. Next — Phase 4: Evaluation Deep-Dive

Phase 4 goes beyond single numbers: calibration curves (are the predicted probabilities *honest*?) and SHAP (which features drive the predictions?). This is where the model's behaviour is understood in depth.
