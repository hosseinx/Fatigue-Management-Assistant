# Fatigue-Management-Assistant
Decision-fatigue check based on your current state
Explainable ML + Streamlit app that recommends Continue / Slow Down / Take Break from a user’s current state (sleep, stress, caffeine, workload, time of day).
The user enters their current state: hours awake, decisions made, sleep last night,
caffeine intake, stress level and time of day. The app returns a recommendation, class
probabilities, and the top factors behind the prediction.

## Highlights
- **No target leakage:** only inputs a real user could report are used.
- **Rigorous model comparison:** 5-fold stratified CV against a baseline, a decision tree,
  logistic regression and random forest. Logistic regression won (macro-F1 ≈ 0.94).
- **Safety-oriented metric:** recall on "Take Break" is tracked, since telling a fatigued
  person to keep working is the costly error.
- **Explainable by design:** per-prediction factor contributions from the linear model.
- **Reproducible pipeline:** shared feature engineering for training and inference,
  saved metrics and confusion matrix, unit tests.
- **Guardrails in the UI:** slider ranges come from the training data, and out-of-range
  inputs trigger a warning.
