# DataPrep-Baseline

Leak-free data cleaning, feature preprocessing and baseline predictive modeling with scikit-learn.

Raw tabular data goes through a single `Pipeline` (`ColumnTransformer` + estimator), so every imputer, scaler and
encoder is fit on training data only. It is evaluated with in-pipeline cross-validation and a hold-out test split.

**Latest release:** [v0.2.0](docs/releases/v0.2.0.md) | [Results on Adult Census Income](docs/results/adult.md)

## Features

- Numeric branch: median imputation, then standard scaling (default) or robust scaling.
- Categorical branch: most-frequent imputation, then one-hot or target encoding (unseen categories are ignored at
  inference).
- Baselines: logistic regression / ridge, random forest, gradient boosting.
- Classification (F1, precision, recall, ROC-AUC) and regression (RMSE, MAE, R2), auto-detected or forced.
- Works with any CSV: pick the target column and optionally drop ID or junk columns with `--drop-cols`.
- Built-in leakage audit, fixed seeds for reproducible runs, and a single serialized `.joblib` pipeline.
- Outputs: scorecard CSV, confusion matrices, ROC curves (or predicted-vs-actual plots), best model.

## Quick start

```bash
pip install -e ".[dev]"
python preprocess_and_model.py                                   # synthetic demo data
python preprocess_and_model.py --data data.csv --target label    # your own CSV
python preprocess_and_model.py --data data.csv --target label --drop-cols id,notes
```

A documented real-world example (Adult Census Income) with download instructions is in
[docs/data/README.md](docs/data/README.md). Any other CSV with a header row and a target column works the same way.

## Documentation

- [Usage](docs/usage.md): CLI flags, environment variables, outputs, using your own dataset
- [Data](docs/data/README.md): example datasets, download steps, dataset requirements
- [Results](docs/results/README.md): measured outputs from real runs
- [Releases](docs/releases/README.md): release notes ([latest: v0.2.0](docs/releases/v0.2.0.md))
- [Architecture](docs/architecture.md): design and data flow
- [Testing](docs/testing.md): test strategy and commands
- [Code quality](docs/code_quality.md): linting, formatting, hooks
- [Roadmap](docs/roadmap.md): milestones and known limitations
- [Contributing](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
