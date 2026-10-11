"""Evaluation. Cross-validation always receives the whole Pipeline (never pre-transformed data)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

LOWER_IS_BETTER = {"rmse", "mae"}


def primary_metric(task: str) -> str:
    return "roc_auc" if task == "classification" else "rmse"


def effective_folds(y_train: np.ndarray, task: str, requested: int) -> int:
    limit = int(np.bincount(y_train).min()) if task == "classification" else len(y_train)
    folds = min(requested, limit)
    if folds < 2:
        raise ValueError("Not enough samples per class/row to run cross-validation.")
    return folds


def _scoring(task: str, n_classes: int) -> dict[str, str]:
    if task == "regression":
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
    if n_classes == 2:
        return {"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    return {
        "f1": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "roc_auc": "roc_auc_ovr_weighted",
    }


def cross_validate_model(
    pipeline: BaseEstimator,
    X_train,
    y_train: np.ndarray,
    task: str,
    n_classes: int,
    folds: int,
    random_state: int,
) -> dict[str, tuple[float, float]]:
    cv_cls = StratifiedKFold if task == "classification" else KFold
    cv = cv_cls(n_splits=folds, shuffle=True, random_state=random_state)
    scoring = _scoring(task, n_classes)
    raw = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
    out: dict[str, tuple[float, float]] = {}
    for metric in scoring:
        values = raw[f"test_{metric}"]
        if metric in LOWER_IS_BETTER:
            values = -values  # sklearn reports errors as negative scores
        out[metric] = (float(values.mean()), float(values.std()))
    return out


def score_holdout(
    fitted: BaseEstimator, X_test, y_test: np.ndarray, task: str, n_classes: int
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    y_pred = fitted.predict(X_test)
    if task == "regression":
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        return metrics, y_pred, None

    proba = fitted.predict_proba(X_test)
    binary = n_classes == 2
    average = "binary" if binary else "weighted"
    auc = (
        roc_auc_score(y_test, proba[:, 1])
        if binary
        else roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
    )
    metrics = {
        "f1": float(f1_score(y_test, y_pred, average=average)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "roc_auc": float(auc),
    }
    return metrics, y_pred, proba
