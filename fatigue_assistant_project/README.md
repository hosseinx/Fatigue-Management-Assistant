# 🧠 Fatigue Management Assistant

A small ML + Streamlit prototype that looks at a user's current state (hours awake, decisions made,
sleep, caffeine, stress, time of day) and recommends **Continue / Slow Down / Take Break**,
with class probabilities and a short explanation of *why*.

> **Honest scope note:** the dataset is synthetic and its labels come from a rule-based fatigue score.
> The model therefore demonstrates a clean, explainable ML pipeline — it is *not* evidence that it predicts
> real human fatigue. Validating on real self-report / behavioural data is the natural next step.

## Project layout
```
fatigue_assistant/   config, features, train, explain, predict   (importable package)
app/                 app.py (Streamlit UI) + style.css
data/                human_decision_fatigue_dataset.xlsx
models/              saved model + metadata (created by training)
reports/             cv_comparison.csv, metrics.json, confusion_matrix.png
tests/               pytest tests
```

## Run
```bash
pip install -r requirements.txt
python -m fatigue_assistant.train     # trains, compares models, saves model + reports
pytest                                # or: python tests/test_model.py
streamlit run app/app.py
```
Re-run training on your machine so the saved model matches your scikit-learn version.

## Method (and why)
- **No target leakage.** Only inputs a real user can report are used. `Error_Rate`, `Fatigue_Level`,
  `Decision_Fatigue_Score` and `Cognitive_Load_Score` are excluded: with them a random forest reaches F1 ≈ 1.00
  because the label is a function of them.
- **Model comparison with 5-fold stratified CV** against a majority-class baseline, a depth-3 tree,
  logistic regression and random forest. Hold-out set is used only for the final report.
- **Metric focus:** macro-F1 plus **recall on “Take Break”** — telling a fatigued person to “Continue”
  is the costly mistake.
- **Logistic regression is selected** (best CV F1 ≈ 0.94, ahead of random forest) because it is as accurate,
  gives well-behaved probabilities, and is directly explainable (per-prediction factor contributions).
- **No train/serve skew:** feature engineering lives inside the saved pipeline.
- **App safeguards:** slider ranges come from training data; out-of-range inputs trigger a warning;
  missing model gives a clear error.

## Results (see `reports/`)
Run training to regenerate; values depend on your environment.
