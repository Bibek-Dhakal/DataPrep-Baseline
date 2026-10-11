# Architecture

```mermaid
flowchart TD
    A[CSV or demo data] --> A2[Drop --drop-cols columns, warn on ID-like columns]
    A2 --> B[Drop rows without target]
    B --> C[Stratified train/test split]
    C --> D[Leakage audit on train/test]
    C --> E[Pipeline: ColumnTransformer + estimator]
    E --> F[Cross-validation on train split]
    E --> G[Fit on train, score on hold-out]
    F --> H[Select best model by CV]
    G --> I[Scorecard, plots, best_model.joblib]
    H --> I
```

## Modules (`src/dataprep_baseline/`)

| Module             | Responsibility                                          |
|--------------------|---------------------------------------------------------|
| `config.py`        | Single source of truth for settings (env vars, validation). |
| `data.py`          | CSV loading, column dropping, ID-like column warnings, synthetic demo data. |
| `preprocessing.py` | Numeric and categorical `ColumnTransformer` branches.   |
| `models.py`        | Baseline estimators and the combined Pipeline factory.  |
| `evaluate.py`      | In-pipeline CV and hold-out metrics.                    |
| `leakage.py`       | Runtime leakage audit.                                  |
| `pipeline.py`      | Orchestration and artifact export.                      |
| `plots.py`         | Confusion matrix, ROC and regression plots.             |
| `cli.py`           | Command line interface.                                 |

## Invariants

- No transformer is fitted outside a `Pipeline`, and none is fitted on hold-out data.
- Cross-validation receives the complete Pipeline, so each fold refits preprocessing on its own training portion.
- Model selection uses CV scores on the training split, never the hold-out set.
- Column dropping is a fixed, data-independent choice made by the user, so it cannot leak information between splits.
- All randomness flows from one seed.
