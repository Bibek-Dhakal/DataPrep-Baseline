# Results: Adult Census Income

Dataset details and download steps are in [../data/README.md](../data/README.md).

## Setup

| Item        | Value                                                                                                               |
|-------------|---------------------------------------------------------------------------------------------------------------------|
| Command     | `python preprocess_and_model.py --data data/adult.csv --target class --drop-cols fnlwgt --output-dir outputs/adult` |
| Task        | Classification (target `class`: `<=50K` / `>50K`)                                                                   |
| Split       | 39,073 train rows, 9,769 hold-out rows (stratified 80/20)                                                           |
| Features    | 5 numeric, 8 categorical (after dropping `fnlwgt`)                                                                  |
| Settings    | Defaults: seed 42, 5-fold CV, `standard` scaler, one-hot encoding                                                   |
| Environment | Windows, Python 3.14.0, run on a laptop                                                                             |
| Runtime     | 12.33 s total                                                                                                       |

## Leakage audit

All three checks passed: `preprocessing_inside_pipeline`, `imputer_stats_equal_train_medians`,
`stats_unchanged_after_holdout_transform`.

## Scorecard

Cross-validation scores (mean and standard deviation over 5 folds, train split only):

| Model               | F1              | Precision       | Recall          | ROC-AUC         |
|---------------------|-----------------|-----------------|-----------------|-----------------|
| gradient_boosting   | 0.7075 ± 0.0074 | 0.7780 ± 0.0110 | 0.6487 ± 0.0079 | 0.9273 ± 0.0026 |
| logistic_regression | 0.6560 ± 0.0095 | 0.7319 ± 0.0176 | 0.5945 ± 0.0075 | 0.9053 ± 0.0040 |
| random_forest       | 0.6535 ± 0.0066 | 0.6985 ± 0.0099 | 0.6142 ± 0.0111 | 0.8887 ± 0.0051 |

Hold-out test scores:

| Model               |     F1 | Precision | Recall | ROC-AUC |
|---------------------|-------:|----------:|-------:|--------:|
| gradient_boosting   | 0.7145 |    0.7880 | 0.6536 |  0.9293 |
| logistic_regression | 0.6581 |    0.7422 | 0.5911 |  0.9039 |
| random_forest       | 0.6613 |    0.7162 | 0.6142 |  0.8915 |

Best model by CV ROC-AUC: **gradient_boosting**. Model selection uses the training split only.

## Plots

![Confusion matrices](../releases/assets/v0.2.0/confusion_matrices_for_adult_csv.jpeg)

![ROC curves](../releases/assets/v0.2.0/roc_curves_for_adult_csv.jpeg)

## Observations

- CV and hold-out ROC-AUC differ by at most 0.003 for every model, consistent with a leak-free setup.
- The classes are imbalanced (about 76% `<=50K` in the hold-out set), so recall for `>50K` is the weakest metric for
  all models.
- Gradient boosting leads on every metric. Logistic regression and random forest are close to each other.
- The default `standard` scaler converged without warnings. With `--scaler robust`, logistic regression emitted
  convergence warnings on this dataset because `capital-gain` and `capital-loss` are mostly zero (see
  [Usage](../usage.md#choosing-a-scaler)).
- Runtime was 12.33 s at about 48k rows, so the original sub-10-second goal is met only for the 2,000-row demo.
  Lower `--cv-folds` or `DATAPREP_RF_ESTIMATORS` to speed up larger runs.
