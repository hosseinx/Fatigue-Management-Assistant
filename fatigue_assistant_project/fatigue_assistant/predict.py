"""Inference helpers used by the app (and by the tests)."""
from functools import lru_cache

import joblib
import pandas as pd

from . import config as C
from .explain import explain_prediction


@lru_cache(maxsize=1)
def load_bundle(path=C.MODEL_PATH) -> dict:
    """Load {'pipeline': ..., 'metadata': ...}; fail with a clear, actionable message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train it first:  python -m fatigue_assistant.train"
        )
    return joblib.load(path)


def validate_inputs(inputs: dict) -> None:
    missing = set(C.RAW_FEATURES) - set(inputs)
    if missing:
        raise ValueError(f"Missing inputs: {sorted(missing)}")
    if inputs["Time_of_Day"] not in C.TIME_OF_DAY_VALUES:
        raise ValueError(f"Time_of_Day must be one of {C.TIME_OF_DAY_VALUES}")


def predict(bundle: dict, inputs: dict, top_k: int = 3) -> dict:
    """Return label, per-class probabilities, top contributing factors, out-of-range warnings."""
    validate_inputs(inputs)
    pipe = bundle["pipeline"]
    X = pd.DataFrame([inputs])[C.RAW_FEATURES]

    proba = pipe.predict_proba(X)[0]
    probs = dict(zip(pipe.classes_, map(float, proba)))
    label = max(probs, key=probs.get)

    warnings = []
    for col, (lo, hi) in bundle["metadata"]["feature_ranges"].items():
        if not lo <= inputs[col] <= hi:
            warnings.append(f"{C.FEATURE_LABELS[col]}={inputs[col]} is outside the training range [{lo:g}, {hi:g}].")

    return {
        "label": label,
        "probabilities": probs,
        "factors": explain_prediction(pipe, X, top_k=top_k),
        "warnings": warnings,
    }
