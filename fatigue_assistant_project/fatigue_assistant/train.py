"""Train, compare and save the fatigue recommendation model.

Run from the project root:
    python -m fatigue_assistant.train
"""
import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .features import add_features

log = logging.getLogger("train")

ENGINEERED_NUMERIC = C.NUMERIC_FEATURES + ["Decisions_per_Hour"]


def load_data() -> pd.DataFrame:
    df = pd.read_excel(C.DATA_PATH)
    missing = set(C.RAW_FEATURES + [C.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    unknown = set(df[C.TARGET].unique()) - set(C.CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unexpected target labels: {unknown}")
    log.info("Loaded %d rows (%d duplicates dropped)", len(df), n_before - len(df))
    return df


def build_pipeline(classifier) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ENGINEERED_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(categories=[C.TIME_OF_DAY_VALUES], handle_unknown="ignore")),
                    ]
                ),
                C.CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def candidates() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=rs),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=rs),
        "Random forest": RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
    }


def compare_models(X, y) -> pd.DataFrame:
    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        # Recall on the costly class: how often a person who needs a break is caught.
        "recall_take_break": make_scorer(recall_score, labels=[C.CRITICAL_CLASS], average="macro"),
    }
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    for name, clf in candidates().items():
        res = cross_validate(build_pipeline(clf), X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for m in scoring:
            vals = res[f"test_{m}"]
            row[f"{m}_mean"], row[f"{m}_std"] = float(vals.mean()), float(vals.std())
        rows.append(row)
        log.info("%-28s F1=%.3f±%.3f  recall(%s)=%.3f", name, row["f1_macro_mean"],
                 row["f1_macro_std"], C.CRITICAL_CLASS, row["recall_take_break_mean"])
    return pd.DataFrame(rows)


def pick_best(cv_table: pd.DataFrame, tolerance: float = 0.01) -> str:
    """Highest CV macro-F1, but prefer the interpretable logistic regression
    when it is within `tolerance` of the best model."""
    real = cv_table[~cv_table["model"].str.startswith("Baseline")]
    best = real.loc[real["f1_macro_mean"].idxmax()]
    lr = real[real["model"] == "Logistic regression"].iloc[0]
    if best["f1_macro_mean"] - lr["f1_macro_mean"] <= tolerance:
        return "Logistic regression"
    return str(best["model"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C.REPORTS_DIR.mkdir(exist_ok=True)
    C.MODEL_PATH.parent.mkdir(exist_ok=True)

    df = load_data()
    X, y = df[C.RAW_FEATURES], df[C.TARGET]

    log.info("\n== 5-fold cross-validation ==")
    cv_table = compare_models(X, y)
    cv_table.to_csv(C.REPORTS_DIR / "cv_comparison.csv", index=False)
    chosen = pick_best(cv_table)
    log.info("\nSelected model: %s", chosen)

    # Held-out evaluation (never used for model selection).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y
    )
    pipe = build_pipeline(candidates()[chosen]).fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, labels=C.CLASS_ORDER, output_dict=True)
    log.info("\n== Hold-out report ==\n%s", classification_report(y_te, y_pred, labels=C.CLASS_ORDER))

    cm = confusion_matrix(y_te, y_pred, labels=C.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASS_ORDER).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Hold-out confusion matrix\n{chosen}")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Deploy: refit on all data. Reported metrics come from CV + the hold-out above.
    final = build_pipeline(candidates()[chosen]).fit(X, y)

    metadata = {
        "model_name": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "classes": list(final.classes_),
        "feature_ranges": {
            c: [float(df[c].min()), float(df[c].max())] for c in C.NUMERIC_FEATURES
        },
        "holdout": {
            "macro_f1": report["macro avg"]["f1-score"],
            "accuracy": report["accuracy"],
            "recall_take_break": report[C.CRITICAL_CLASS]["recall"],
        },
        "cv": cv_table.set_index("model").loc[chosen].to_dict(),
    }
    joblib.dump({"pipeline": final, "metadata": metadata}, C.MODEL_PATH)
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved model -> %s", C.MODEL_PATH)


if __name__ == "__main__":
    main()
