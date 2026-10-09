"""Feature engineering shared by training and inference (no train/serve skew)."""
import pandas as pd

from .config import RAW_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Input must contain the raw feature columns."""
    out = df[RAW_FEATURES].copy()
    # Decisions made per hour awake (+1 avoids division by zero right after waking).
    out["Decisions_per_Hour"] = out["Decisions_Made"] / (out["Hours_Awake"] + 1)
    return out
