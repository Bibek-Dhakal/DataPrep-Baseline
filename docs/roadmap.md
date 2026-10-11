# Roadmap

## Current (v0.2)

- Leak-free Pipeline with classification and regression baselines, audit, scorecard and plots.
- Bring-your-own CSV with `--drop-cols` and ID-like column warnings.
- Standard scaling by default, robust scaling optional; configurable logistic regression iteration limit.
- Verified on Adult Census Income ([results](results/adult.md)).

## Planned

- Optional LightGBM / XGBoost baselines.
- Time-aware splitting for temporal data (`TimeSeriesSplit`).
- Optional hyperparameter search inside the Pipeline.
- Model calibration plots.
- Automatic coercion of numeric-looking text columns to numeric.
- Declaring numeric-coded columns as categorical (`--categorical-cols`).
- Basic date feature extraction.
- Outlier-aware scaling that also handles zero-inflated columns (for example a quantile or log transform step).
- Faster large-dataset runs (lighter default random forest or optional subsampling).

## Known limitations

- Task detection is heuristic (override with `--task`).
- Numbers stored as text are treated as categorical; dates are not parsed.
- ID-column detection is a warning heuristic; columns to exclude must be passed through `--drop-cols`.
- The default `standard` scaler is sensitive to extreme outliers; `robust` resists them but cannot scale zero-inflated
  columns (interquartile range of 0).
- Runtime grows with dataset size: seconds on the 2,000-row demo, about 12 seconds on the 48k-row Adult dataset on a
  Windows laptop, depending on hardware and `DATAPREP_RF_ESTIMATORS`. See [usage.md](usage.md#runtime) for speed-up
  options.
