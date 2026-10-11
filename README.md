# DataPrep-Baseline

Leak-free data cleaning, feature preprocessing and baseline predictive modeling with scikit-learn.

Raw tabular data goes through a single `Pipeline` (`ColumnTransformer` + estimator), so every imputer, scaler and
encoder is fit on training data only. It is evaluated with in-pipeline cross-validation and a hold-out test split.

## Features

- Numeric branch: median imputation, then robust or standard scaling.
- Categorical branch: most-frequent imputation, then one-hot or target encoding (unseen categories are ignored at
  inference).
- Baselines: logistic regression / ridge, random forest, gradient boosting.
- Classification (F1, precision, recall, ROC-AUC) and regression (RMSE, MAE, R2), auto-detected or forced.
- Built-in leakage audit, fixed seeds for reproducible runs, and a single serialized `.joblib` pipeline.
- Outputs: scorecard CSV, confusion matrices, ROC curves (or predicted-vs-actual plots), best model.

## Quick start

```bash
pip install -e ".[dev]"
python preprocess_and_model.py                                   # synthetic demo data
python preprocess_and_model.py --data data.csv --target label    # your CSV
```

## Documentation

- [Usage](docs/usage.md): CLI flags, environment variables, outputs
- [Architecture](docs/architecture.md): design and data flow
- [Testing](docs/testing.md): test strategy and commands
- [Code quality](docs/code_quality.md): linting, formatting, hooks
- [Roadmap](docs/roadmap.md): milestones and known limitations
- [Contributing](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
