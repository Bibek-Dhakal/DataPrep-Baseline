# Usage

## Commands

```bash
python preprocess_and_model.py [options]
dataprep-baseline [options]
python -m dataprep_baseline [options]
```

| Flag           | Description                                                             |
|----------------|-------------------------------------------------------------------------|
| `--data`       | Input CSV path. Omit to use the synthetic demo data.                    |
| `--target`     | Target column (required with `--data`).                                 |
| `--drop-cols`  | Comma-separated feature columns to drop, e.g. `id,notes`.               |
| `--task`       | `auto`, `classification` or `regression`.                               |
| `--seed`       | Random seed.                                                            |
| `--test-size`  | Hold-out fraction (0-1).                                                |
| `--cv-folds`   | Cross-validation folds (reduced automatically if a class is too small). |
| `--scaler`     | `standard` (default) or `robust`.                                       |
| `--encoder`    | `onehot` or `target`.                                                   |
| `--output-dir` | Artifact directory.                                                     |

Task auto-detection: a non-numeric target, or a numeric target with 10 or fewer unique values, is classification;
anything else is regression.

Invalid input (missing file, missing target, unknown or target column in `--drop-cols`, invalid settings) prints a
clear error and exits with code 2.

## Using your own dataset

```bash
python preprocess_and_model.py --data path/to/data.csv --target your_label
python preprocess_and_model.py --data path/to/data.csv --target price --task regression --drop-cols id
```

Requirements, ID-column guidance and a documented example dataset are in [data/README.md](data/README.md).

## Choosing a scaler

| Scaler     | Behaviour                                                                                                                                                           |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `standard` | Default. Centers on the mean and scales by standard deviation. Works on zero-inflated skewed columns, but is more sensitive to extreme outliers.                    |
| `robust`   | Centers on the median and scales by the interquartile range. Resists outliers, but leaves columns unscaled when the interquartile range is 0 (mostly-zero columns). |

If your data has extreme outliers, compare both scalers: `--scaler robust` and `--scaler standard`.

## Runtime

Runtime depends on row count, column count, hardware and the random forest size. The synthetic demo (2,000 rows)
finishes in a few seconds. The documented Adult run (about 48k rows) took about 12 seconds with default settings on a
Windows laptop (see [results](results/adult.md)), mostly from random forest with cross-validation. To speed up large
runs:

```bash
python preprocess_and_model.py --data data/big.csv --target label --cv-folds 3
DATAPREP_RF_ESTIMATORS=50 python preprocess_and_model.py --data data/big.csv --target label
```

On PowerShell, set the variable first: `$env:DATAPREP_RF_ESTIMATORS="50"`.

## Convergence warnings

If logistic regression prints `ConvergenceWarning`, its optimizer hit the iteration limit. This is almost always a
feature scaling issue:

- With `--scaler robust`, zero-inflated skewed columns stay unscaled. Use `--scaler standard` (the default).
- Raising `DATAPREP_LR_MAX_ITER` rarely fixes this and makes runs much slower. Fix the scaling first.

## Environment variables

Defaults live in `src/dataprep_baseline/config.py`. CLI flags override environment variables.

| Name                     | Type   | Default      | Description                              |
|--------------------------|--------|--------------|------------------------------------------|
| `DATAPREP_DATA_PATH`     | path   | empty (demo) | Input CSV path.                          |
| `DATAPREP_TARGET`        | string | empty        | Target column name.                      |
| `DATAPREP_DROP_COLS`     | string | empty        | Comma-separated feature columns to drop. |
| `DATAPREP_TASK`          | string | `auto`       | `auto`, `classification`, `regression`.  |
| `DATAPREP_RANDOM_STATE`  | int    | `42`         | Seed for splits, CV and models.          |
| `DATAPREP_TEST_SIZE`     | float  | `0.2`        | Hold-out fraction.                       |
| `DATAPREP_CV_FOLDS`      | int    | `5`          | Cross-validation folds.                  |
| `DATAPREP_SCALER`        | string | `standard`   | `standard` or `robust`.                  |
| `DATAPREP_ENCODER`       | string | `onehot`     | `onehot` or `target`.                    |
| `DATAPREP_N_JOBS`        | int    | `-1`         | Parallel jobs for random forest.         |
| `DATAPREP_RF_ESTIMATORS` | int    | `100`        | Trees in random forest.                  |
| `DATAPREP_LR_MAX_ITER`   | int    | `1000`       | Iteration limit for logistic regression. |
| `DATAPREP_OUTPUT_DIR`    | path   | `outputs`    | Artifact directory.                      |

## Outputs

| File                                       | Content                                                     |
|--------------------------------------------|-------------------------------------------------------------|
| `scorecard.csv`                            | CV mean/std and hold-out metrics per model.                 |
| `best_model.joblib`                        | Best model (selected by CV on train split) as one Pipeline. |
| `confusion_matrices.png`, `roc_curves.png` | Classification plots.                                       |
| `predicted_vs_actual.png`                  | Regression plot.                                            |

Load a saved model with `joblib.load("outputs/best_model.joblib")` and call `.predict()` on a raw feature DataFrame
with the same columns used for training (after any `--drop-cols` columns are removed).
