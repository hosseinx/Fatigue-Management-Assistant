"""Central configuration: paths, feature lists, labels.

Everything the training script and the app must agree on lives here,
so the two can never silently drift apart.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "human_decision_fatigue_dataset.xlsx"
MODEL_PATH = ROOT / "models" / "break_assistant_model.joblib"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "System_Recommendation"

# Only inputs a real user could plausibly report.
# (Error_Rate, Fatigue_Level, Decision_Fatigue_Score, Cognitive_Load_Score ... are
#  derived from / used to build the label -> using them would be target leakage.)
NUMERIC_FEATURES = [
    "Hours_Awake",
    "Decisions_Made",
    "Sleep_Hours_Last_Night",
    "Caffeine_Intake_Cups",
    "Stress_Level_1_10",
]
CATEGORICAL_FEATURES = ["Time_of_Day"]
TIME_OF_DAY_VALUES = ["Morning", "Afternoon", "Evening", "Night"]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Order matters: least -> most severe.
CLASS_ORDER = ["Continue", "Slow Down", "Take Break"]
# The costly mistake is telling a fatigued person to "Continue".
CRITICAL_CLASS = "Take Break"

FEATURE_LABELS = {
    "Hours_Awake": "Hours awake",
    "Decisions_Made": "Decisions made",
    "Sleep_Hours_Last_Night": "Sleep last night",
    "Caffeine_Intake_Cups": "Caffeine intake",
    "Stress_Level_1_10": "Stress level",
    "Decisions_per_Hour": "Decision density",
    "Time_of_Day": "Time of day",
}
