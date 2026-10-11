"""End-to-end orchestration: split -> audit -> CV on train -> hold-out verification -> export."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import Config
from .evaluate import (
    LOWER_IS_BETTER,
    cross_validate_model,
    effective_folds,
    primary_metric,
    score_holdout,
)
from .leakage import AuditCheck, audit_no_leakage
from .models import build_pipeline, get_models
from .plots import plot_confusion_matrices, plot_predicted_vs_actual, plot_roc_curves
from .preprocessing import build_preprocessor, split_columns

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    pipeline: BaseEstimator
    cv: dict[str, tuple[float, float]]
    test: dict[str, float]
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    fit_seconds: float


@dataclass
class ExperimentResult:
    task: str
    scorecard: pd.DataFrame
    models: list[ModelResult]
    y_test: np.ndarray
    class_names: list[str]
    best_name: str
    audit: list[AuditCheck] = field(default_factory=list)


def detect_task(y: pd.Series, requested: str) -> str:
    if requested != "auto":
        return requested
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    return "classification" if y.nunique() <= 10 else "regression"


def run_experiment(X: pd.DataFrame, y: pd.Series, cfg: Config) -> ExperimentResult:
    keep = y.notna().to_numpy()  # rows without a target cannot be used for supervised learning
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

    task = detect_task(y, cfg.task)
    class_names: list[str] = []
    if task == "classification":
        label_encoder = LabelEncoder()
        y_arr = label_encoder.fit_transform(y)
        class_names = [str(c) for c in label_encoder.classes_]
        if len(class_names) < 2:
            raise ValueError("Classification needs at least two classes in the target.")
        stratify = y_arr
    else:
        y_arr = y.astype(float).to_numpy()
        stratify = None
    n_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify
    )
    numeric, categorical = split_columns(X_train)
    preprocessor = build_preprocessor(
        numeric, categorical, cfg.scaler, cfg.categorical_encoder, cfg.random_state
    )
    estimators = get_models(task, cfg)
    folds = effective_folds(y_train, task, cfg.cv_folds)
    logger.info(
        "Task: %s | train=%d test=%d | numeric=%d categorical=%d | cv_folds=%d",
        task,
        len(X_train),
        len(X_test),
        len(numeric),
        len(categorical),
        folds,
    )

    audit = audit_no_leakage(
        build_pipeline(preprocessor, next(iter(estimators.values()))),
        X_train,
        y_train,
        X_test,
        numeric,
    )
    for check in audit:
        logger.info("Leakage audit [%s] %s", "PASS" if check.passed else "FAIL", check.name)
    if not all(c.passed for c in audit):
        raise RuntimeError("Leakage audit failed; refusing to report scores.")

    results: list[ModelResult] = []
    for name, estimator in estimators.items():
        pipe = build_pipeline(preprocessor, estimator)
        cv_scores = cross_validate_model(
            pipe, X_train, y_train, task, n_classes, folds, cfg.random_state
        )
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        test_scores, y_pred, y_proba = score_holdout(pipe, X_test, y_test, task, n_classes)
        results.append(
            ModelResult(name, pipe, cv_scores, test_scores, y_pred, y_proba, fit_seconds)
        )

    rows = []
    for r in results:
        row: dict[str, float | str] = {"model": r.name}
        for metric, (mean, std) in r.cv.items():
            row[f"cv_{metric}_mean"] = mean
            row[f"cv_{metric}_std"] = std
        for metric, value in r.test.items():
            row[f"test_{metric}"] = value
        rows.append(row)

    main_metric = primary_metric(task)
    sort_col = f"cv_{main_metric}_mean"
    ascending = main_metric in LOWER_IS_BETTER
    scorecard = pd.DataFrame(rows).set_index("model").sort_values(sort_col, ascending=ascending)
    # Model selection uses CV on the training split only, never the hold-out test split.
    best_name = str(scorecard.index[0])

    return ExperimentResult(task, scorecard, results, y_test, class_names, best_name, audit)


def export_artifacts(result: ExperimentResult, cfg: Config) -> list[Path]:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    scorecard_path = out / "scorecard.csv"
    result.scorecard.to_csv(scorecard_path, float_format="%.4f")
    paths.append(scorecard_path)

    best = next(m for m in result.models if m.name == result.best_name)
    model_path = out / "best_model.joblib"
    joblib.dump(best.pipeline, model_path)
    paths.append(model_path)

    if result.task == "classification":
        cm_path = out / "confusion_matrices.png"
        roc_path = out / "roc_curves.png"
        plot_confusion_matrices(result.models, result.y_test, result.class_names, cm_path)
        plot_roc_curves(result.models, result.y_test, result.class_names, roc_path)
        paths += [cm_path, roc_path]
    else:
        pv_path = out / "predicted_vs_actual.png"
        plot_predicted_vs_actual(result.models, result.y_test, pv_path)
        paths.append(pv_path)
    return paths
