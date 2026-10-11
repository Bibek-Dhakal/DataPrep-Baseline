"""Baseline estimators per task and the single serializable Pipeline factory."""

from __future__ import annotations

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .config import Config


def get_models(task: str, cfg: Config) -> dict[str, BaseEstimator]:
    rs = cfg.random_state
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=rs),
            "random_forest": RandomForestClassifier(
                n_estimators=cfg.rf_estimators, n_jobs=cfg.n_jobs, random_state=rs
            ),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=rs),
        }
    return {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(
            n_estimators=cfg.rf_estimators, n_jobs=cfg.n_jobs, random_state=rs
        ),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=rs),
    }


def build_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline:
    """Preprocessing and estimator in one object so every fit is isolated per split."""
    return Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
