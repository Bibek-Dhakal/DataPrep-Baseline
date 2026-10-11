"""Preprocessing builders. Transformers are only declared here; fitting happens inside Pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    scaler: str,
    encoder: str,
    random_state: int,
) -> ColumnTransformer:
    scale = RobustScaler() if scaler == "robust" else StandardScaler()
    numeric_branch = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scale)])

    if encoder == "target":
        encode = TargetEncoder(random_state=random_state)
    else:
        encode = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_branch = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encode)]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_branch, numeric), ("cat", categorical_branch, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
