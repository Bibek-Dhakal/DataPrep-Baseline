import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from dataprep_baseline.cli import main
from dataprep_baseline.config import Config, parse_columns
from dataprep_baseline.data import load_dataset, make_demo_dataset
from dataprep_baseline.leakage import audit_no_leakage
from dataprep_baseline.models import build_pipeline, get_models
from dataprep_baseline.pipeline import run_experiment
from dataprep_baseline.preprocessing import build_preprocessor, split_columns

CFG = Config(random_state=0, cv_folds=3, rf_estimators=10)


class FitSpy(BaseEstimator, TransformerMixin):
    """Records how many rows every fit() call sees."""

    seen: list[int] = []

    def fit(self, X, y=None):
        FitSpy.seen.append(len(X))
        return self

    def transform(self, X):
        return X


@pytest.fixture
def split():
    X, y = make_demo_dataset(0, n_rows=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y.to_numpy(), test_size=0.2, random_state=0, stratify=y
    )
    numeric, categorical = split_columns(X_tr)
    pre = build_preprocessor(numeric, categorical, CFG.scaler, CFG.categorical_encoder, 0)
    return X_tr, X_te, y_tr, y_te, numeric, pre


@pytest.fixture
def custom_csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "row_id": range(200),
            "x": rng.normal(size=200),
            "grp": rng.choice(["a", "b"], 200),
            "label": rng.integers(0, 2, 200),
        }
    )
    path = tmp_path / "custom.csv"
    df.to_csv(path, index=False)
    return path


def test_transformers_never_fit_on_holdout(split):
    X_tr, X_te, y_tr, _, _, pre = split
    model = get_models("classification", CFG)["logistic_regression"]
    pipe = Pipeline([("spy", FitSpy()), ("preprocess", pre), ("model", model)])

    FitSpy.seen.clear()
    cross_validate(pipe, X_tr, y_tr, cv=3)
    assert FitSpy.seen and max(FitSpy.seen) < len(X_tr)  # folds fit on subsets of train only

    FitSpy.seen.clear()
    pipe.fit(X_tr, y_tr)
    pipe.predict(X_te)
    assert FitSpy.seen == [len(X_tr)]  # predict() never triggers a fit


def test_leakage_audit_passes(split):
    X_tr, X_te, y_tr, _, numeric, pre = split
    model = get_models("classification", CFG)["random_forest"]
    checks = audit_no_leakage(build_pipeline(pre, model), X_tr, y_tr, X_te, numeric)
    assert checks and all(c.passed for c in checks)


def test_unseen_category_at_inference(split):
    X_tr, X_te, y_tr, _, _, pre = split
    model = get_models("classification", CFG)["logistic_regression"]
    pipe = build_pipeline(pre, model).fit(X_tr, y_tr)
    X_new = X_te.copy()
    X_new["region"] = "brand_new_region"
    assert len(pipe.predict(X_new)) == len(X_new)


def test_reproducible_across_runs():
    X, y = make_demo_dataset(3, n_rows=300)
    a = run_experiment(X, y, CFG).scorecard
    b = run_experiment(X, y, CFG).scorecard
    pd.testing.assert_frame_equal(a, b)


def test_regression_task():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    X.loc[::17, "a"] = np.nan
    y = pd.Series(2 * X["b"].to_numpy() + rng.normal(0, 0.1, 300))
    result = run_experiment(X, y, CFG)
    assert result.task == "regression"
    assert "test_rmse" in result.scorecard.columns


def test_cli_requires_target_with_data(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    assert main(["--data", str(csv)]) == 2


def test_cli_demo_run_exports_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAPREP_RF_ESTIMATORS", "10")
    assert main(["--cv-folds", "3", "--output-dir", str(tmp_path)]) == 0
    for name in ("scorecard.csv", "best_model.joblib", "confusion_matrices.png", "roc_curves.png"):
        assert (tmp_path / name).exists()


def test_parse_columns():
    assert parse_columns("a, b,,c ") == ("a", "b", "c")
    assert parse_columns("") == ()


def test_drop_cols_removes_features(custom_csv):
    X, y = load_dataset(custom_csv, "label", 0, ("row_id",))
    assert list(X.columns) == ["x", "grp"]
    assert len(X) == len(y)


def test_drop_cols_rejects_target_and_unknown_columns(custom_csv):
    with pytest.raises(ValueError, match="target"):
        load_dataset(custom_csv, "label", 0, ("label",))
    with pytest.raises(ValueError, match="not found"):
        load_dataset(custom_csv, "label", 0, ("nope",))


def test_id_like_column_warns(custom_csv, caplog):
    with caplog.at_level(logging.WARNING, logger="dataprep_baseline.data"):
        load_dataset(custom_csv, "label", 0)
    assert "row_id" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dataprep_baseline.data"):
        load_dataset(custom_csv, "label", 0, ("row_id",))
    assert "row_id" not in caplog.text


def test_cli_missing_file_exits_with_error(tmp_path):
    assert main(["--data", str(tmp_path / "missing.csv"), "--target", "label"]) == 2


def test_cli_custom_csv_with_drop_cols(custom_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("DATAPREP_RF_ESTIMATORS", "10")
    out = tmp_path / "out"
    argv = ["--data", str(custom_csv), "--target", "label", "--drop-cols", "row_id"]
    assert main([*argv, "--cv-folds", "3", "--output-dir", str(out)]) == 0
    assert (out / "scorecard.csv").exists()
