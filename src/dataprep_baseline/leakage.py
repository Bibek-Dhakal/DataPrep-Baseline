"""Runtime audit that fitted preprocessing statistics come from training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


def audit_no_leakage(
    pipeline: BaseEstimator, X_train, y_train, X_test, numeric_cols: list[str]
) -> list[AuditCheck]:
    fitted = clone(pipeline).fit(X_train, y_train)
    pre = fitted.named_steps["preprocess"]
    checks = [
        AuditCheck(
            "preprocessing_inside_pipeline",
            "preprocess" in fitted.named_steps,
            "All transformers are steps of the same Pipeline as the estimator.",
        )
    ]
    if numeric_cols:
        imputer = pre.named_transformers_["num"].named_steps["imputer"]
        expected = X_train[numeric_cols].median().to_numpy(dtype=float)
        checks.append(
            AuditCheck(
                "imputer_stats_equal_train_medians",
                bool(np.allclose(imputer.statistics_, expected, equal_nan=True)),
                "Fitted median imputation values equal medians of the training split.",
            )
        )
        before = imputer.statistics_.copy()
        fitted.predict(X_test)
        checks.append(
            AuditCheck(
                "stats_unchanged_after_holdout_transform",
                bool(np.array_equal(before, imputer.statistics_, equal_nan=True)),
                "Transforming hold-out data does not refit or alter fitted statistics.",
            )
        )
    return checks
