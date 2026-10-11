"""Dataset ingestion. No fitting or statistics are computed here (leak-safe by design)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_demo_dataset(random_state: int = 42, n_rows: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic churn-style data with missing values, outliers and categoricals."""
    rng = np.random.default_rng(random_state)
    age = rng.normal(40, 12, n_rows).round()
    income = rng.lognormal(10.5, 0.6, n_rows)
    income[rng.choice(n_rows, size=max(1, n_rows // 100), replace=False)] *= 20  # outliers
    tenure = rng.integers(0, 120, n_rows).astype(float)
    plan = rng.choice(["basic", "pro", "enterprise"], n_rows, p=[0.6, 0.3, 0.1])
    region = rng.choice(["north", "south", "east", "west"], n_rows)

    logit = (
        0.8
        - 0.025 * tenure
        + 0.02 * (40 - age)
        + 0.8 * (plan == "basic")
        - 0.8 * (plan == "enterprise")
        + rng.normal(0, 0.5, n_rows)
    )
    y = pd.Series((rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int), name="churn")

    X = pd.DataFrame(
        {"age": age, "income": income, "tenure": tenure, "plan": plan, "region": region}
    )
    for col in X.columns:
        X.loc[rng.random(n_rows) < 0.08, col] = np.nan
    return X, y


def load_dataset(
    path: Path | None, target: str | None, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV (features, target) or fall back to the synthetic demo dataset."""
    if path is None:
        logger.info("No dataset supplied; using synthetic demo data (target: churn).")
        return make_demo_dataset(random_state)
    if not target:
        raise ValueError("A target column is required when a dataset is supplied (--target).")
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
    return df.drop(columns=[target]), df[target]
