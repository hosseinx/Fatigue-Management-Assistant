import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fatigue_assistant import config as C
from fatigue_assistant.features import add_features
from fatigue_assistant.predict import load_bundle, predict, validate_inputs

RESTED = dict(Hours_Awake=2, Decisions_Made=5, Sleep_Hours_Last_Night=8.0,
              Caffeine_Intake_Cups=0, Stress_Level_1_10=1, Time_of_Day="Morning")
EXHAUSTED = dict(Hours_Awake=16, Decisions_Made=100, Sleep_Hours_Last_Night=4.0,
                 Caffeine_Intake_Cups=5, Stress_Level_1_10=7, Time_of_Day="Night")


def test_add_features_decisions_per_hour():
    df = pd.DataFrame([RESTED])
    out = add_features(df)
    assert out.loc[0, "Decisions_per_Hour"] == 5 / 3


def test_prediction_is_valid_class_and_probs_sum_to_one():
    res = predict(load_bundle(), RESTED)
    assert res["label"] in C.CLASS_ORDER
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-6


def test_extreme_cases_behave_sensibly():
    b = load_bundle()
    assert predict(b, RESTED)["label"] == "Continue"
    assert predict(b, EXHAUSTED)["label"] == "Take Break"


def test_missing_input_raises():
    bad = {k: v for k, v in RESTED.items() if k != "Hours_Awake"}
    try:
        validate_inputs(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_invalid_time_of_day_raises():
    try:
        validate_inputs({**RESTED, "Time_of_Day": "Noon"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_out_of_range_input_warns():
    res = predict(load_bundle(), {**RESTED, "Hours_Awake": 24})
    assert res["warnings"]


def test_explanations_returned():
    res = predict(load_bundle(), EXHAUSTED, top_k=3)
    assert len(res["factors"]) == 3
    assert all(isinstance(v, float) for _, v in res["factors"])


if __name__ == "__main__":  # allows running without pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
