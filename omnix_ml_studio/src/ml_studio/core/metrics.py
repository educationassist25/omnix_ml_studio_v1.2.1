"""Metric definitions with explicit averaging, used for CV, holdout and external data.

* Binary: positive class = encoded 1. ROC-AUC and average precision use P(class 1).
* Multiclass: ROC-AUC is one-vs-rest, macro-averaged; precision/recall/F1 are
  macro-averaged (each class counts equally regardless of size).
* Regression: R², RMSE, MAE, MSE, median absolute error; MAPE only if all |y| > 0.
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M
from sklearn.metrics import make_scorer

CLASSIFICATION_PRIMARY = {"binary": "roc_auc", "multiclass": "roc_auc_ovr"}
REGRESSION_PRIMARY = "r2"

METRIC_LABELS = {
    "roc_auc": "ROC-AUC", "roc_auc_ovr": "ROC-AUC (OvR macro)", "average_precision": "PR-AUC (AP)",
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision", "recall": "Recall", "f1": "F1",
    "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)",
    "mcc": "MCC", "log_loss": "Log loss",
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "mse": "MSE", "medae": "Median AE", "mape": "MAPE",
}
LOWER_IS_BETTER = {"log_loss", "rmse", "mae", "mse", "medae", "mape"}


def _rmse(y, p):
    return float(np.sqrt(M.mean_squared_error(y, p)))


def scorers(task: str, n_classes: int = 2, y=None) -> dict:
    """sklearn scorer dict. Error metrics are returned *negated* by sklearn conventions;
    :func:`tidy_cv` flips them back."""
    labels = list(range(n_classes))
    if task == "binary":
        return {
            "roc_auc": "roc_auc", "average_precision": "average_precision",
            "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
            "precision": make_scorer(M.precision_score, zero_division=0),
            "recall": "recall", "f1": make_scorer(M.f1_score, zero_division=0),
            "mcc": make_scorer(M.matthews_corrcoef),
            "log_loss": make_scorer(M.log_loss, greater_is_better=False, response_method="predict_proba",
                                    labels=labels),
        }
    if task == "multiclass":
        return {
            "roc_auc_ovr": make_scorer(M.roc_auc_score, response_method="predict_proba",
                                       multi_class="ovr", average="macro", labels=labels),
            "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
            "precision_macro": make_scorer(M.precision_score, average="macro", zero_division=0),
            "recall_macro": make_scorer(M.recall_score, average="macro", zero_division=0),
            "f1_macro": make_scorer(M.f1_score, average="macro", zero_division=0),
            "mcc": make_scorer(M.matthews_corrcoef),
            "log_loss": make_scorer(M.log_loss, greater_is_better=False, response_method="predict_proba",
                                    labels=labels),
        }
    out = {"r2": "r2", "rmse": make_scorer(_rmse, greater_is_better=False),
           "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error",
           "medae": "neg_median_absolute_error"}
    if y is not None and np.all(np.abs(np.asarray(y)) > 1e-8):
        out["mape"] = "neg_mean_absolute_percentage_error"
    return out


def has_proba(model) -> bool:
    """True if the (unfitted or fitted) estimator / pipeline exposes calibrated-ish probabilities."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    try:
        return hasattr(est, "predict_proba")
    except Exception:  # noqa: BLE001
        return False


def scorers_for(task: str, n_classes: int, y, proba: bool) -> dict:
    """Scorer dict usable by a model. Without probabilities: no log loss; binary ROC-AUC /
    average precision fall back to the decision function; multiclass ROC-AUC is unavailable."""
    sc = scorers(task, n_classes, y)
    if not proba and task != "regression":
        sc.pop("log_loss", None)
        if task == "multiclass":
            sc.pop("roc_auc_ovr", None)
    return sc


def primary_metric(task: str) -> str:
    return REGRESSION_PRIMARY if task == "regression" else CLASSIFICATION_PRIMARY[task]


def compute(task: str, y_true, y_pred=None, proba=None, n_classes: int = 2, calibrated: bool = True) -> dict:
    """Metrics from predictions (holdout / external / bootstrap).

    ``calibrated=False`` means ``proba`` holds monotone scores (e.g. a sigmoid of the decision
    function): rank metrics (ROC-AUC, AP) are valid, log loss is not reported."""
    y_true = np.asarray(y_true)
    out = {}
    if task == "regression":
        p = np.asarray(y_pred, dtype=float)
        out = {"r2": M.r2_score(y_true, p), "rmse": _rmse(y_true, p),
               "mae": M.mean_absolute_error(y_true, p), "mse": M.mean_squared_error(y_true, p),
               "medae": M.median_absolute_error(y_true, p)}
        if np.all(np.abs(y_true) > 1e-8):
            out["mape"] = M.mean_absolute_percentage_error(y_true, p)
        return out
    labels = list(range(n_classes))
    if y_pred is None:
        y_pred = np.argmax(proba, axis=1)
    out["accuracy"] = M.accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = M.balanced_accuracy_score(y_true, y_pred)
    out["mcc"] = M.matthews_corrcoef(y_true, y_pred)
    present = np.unique(y_true)
    if task == "binary":
        out["precision"] = M.precision_score(y_true, y_pred, zero_division=0)
        out["recall"] = M.recall_score(y_true, y_pred, zero_division=0)
        out["f1"] = M.f1_score(y_true, y_pred, zero_division=0)
        if proba is not None and len(present) == 2:
            out["roc_auc"] = M.roc_auc_score(y_true, proba[:, 1])
            out["average_precision"] = M.average_precision_score(y_true, proba[:, 1])
    else:
        for k, f in (("precision_macro", M.precision_score), ("recall_macro", M.recall_score),
                     ("f1_macro", M.f1_score)):
            out[k] = f(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        if proba is not None and len(present) == n_classes:
            out["roc_auc_ovr"] = M.roc_auc_score(y_true, proba, multi_class="ovr",
                                                 average="macro", labels=labels)
    if proba is not None and calibrated:
        out["log_loss"] = M.log_loss(y_true, np.clip(proba, 1e-15, 1), labels=labels)
    return out
