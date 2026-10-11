# Usage

## Commands

```bash
python preprocess_and_model.py [options]
dataprep-baseline [options]
python -m dataprep_baseline [options]
```

| Flag           | Description                                          |
|----------------|------------------------------------------------------|
| `--data`       | Input CSV path. Omit to use the synthetic demo data. |
| `--target`     | Target column (required with `--data`).              |
| `--drop-cols`  | Comma-separated feature columns to drop, e.g. `id,notes`. |
| `--task`       | `auto`, `classification` or `regression`.            |
| `--seed`       | Random seed.                                         |
| `--test-size`  | Hold-out fraction (0-1).                             |
| `--cv-folds`   | Cross-validation folds (reduced automatically if a class is too small). |
| `--scaler`     | `robust` or `standard`.                              |
| `--encoder`    | `onehot` or `target`.                                |
| `--output-dir` | Artifact directory.                                  |

Task auto-detection: a non-numeric target, or a numeric target with 10 or fewer unique values, is classification;
anything else is regression.

Invalid input (missing file, missing target, unknown or target column in `--drop-cols`) prints a clear error and exits
with code 2.

## Using your own dataset

```bash
python preprocess_and_model.py --data path/to/data.csv --target your_label
python preprocess_and_model.py --data path/to/data.csv --target price --task regression --drop-cols id
```

Requirements, ID-column guidance and a documented example dataset are in [data/README.md](data/README.md).

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
| `DATAPREP_SCALER`        | string | `robust`     | `robust` or `standard`.                  |
| `DATAPREP_ENCODER`       | string | `onehot`     | `onehot` or `target`.                    |
| `DATAPREP_N_JOBS`        | int    | `-1`         | Parallel jobs for random forest.         |
| `DATAPREP_RF_ESTIMATORS` | int    | `100`        | Trees in random forest.                  |
| `DATAPREP_OUTPUT_DIR`    | path   | `outputs`    | Artifact directory.                      |

## Outputs

| File                                   | Content                                         |
|----------------------------------------|-------------------------------------------------|
| `scorecard.csv`                        | CV mean/std and hold-out metrics per model.     |
| `best_model.joblib`                    | Best model (selected by CV on train split) as one Pipeline. |
| `confusion_matrices.png`, `roc_curves.png` | Classification plots.                       |
| `predicted_vs_actual.png`              | Regression plot.                                |

Load a saved model with `joblib.load("outputs/best_model.joblib")` and call `.predict()` on a raw feature DataFrame
with the same columns used for training (after any `--drop-cols` columns are removed).
