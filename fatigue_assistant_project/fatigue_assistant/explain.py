"""Per-prediction explanations for the (multinomial) logistic regression model.

For a linear softmax model the logit of class c is  b_c + sum_j w_cj * x_j.
Subtracting the mean logit over classes does not change the probabilities, so
    contribution_j = w_{pred,j} * x_j  -  mean_c(w_cj * x_j)
tells how much feature j pushed the model toward the predicted class.
"""
import numpy as np
import pandas as pd

from .config import FEATURE_LABELS


def _group_name(transformed_name: str) -> str:
    # 'num__Hours_Awake' -> 'Hours_Awake' ; 'cat__Time_of_Day_Night' -> 'Time_of_Day'
    name = transformed_name.split("__", 1)[1]
    return "Time_of_Day" if name.startswith("Time_of_Day") else name


def explain_prediction(pipeline, input_df: pd.DataFrame, top_k: int = 3):
    """Return [(label, contribution), ...] sorted by |contribution|, for the predicted class.

    Positive contribution = pushed toward the predicted class.
    Returns an empty list if the final estimator is not linear.
    """
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "coef_"):
        return []

    prep = pipeline[:-1]
    x = prep.transform(input_df)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    names = prep.named_steps["preprocessor"].get_feature_names_out()

    pred_idx = int(np.argmax(clf.predict_proba(prep.transform(input_df))[0]))
    per_class = clf.coef_ * x  # (n_classes, n_features)
    contrib = per_class[pred_idx] - per_class.mean(axis=0)

    grouped: dict[str, float] = {}
    for n, c in zip(names, contrib):
        g = _group_name(n)
        grouped[g] = grouped.get(g, 0.0) + float(c)

    ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return [(FEATURE_LABELS.get(k, k), v) for k, v in ranked]
