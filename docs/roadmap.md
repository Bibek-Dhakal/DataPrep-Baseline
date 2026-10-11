# Roadmap

## Current (v0.1)

- Leak-free Pipeline with classification and regression baselines, audit, scorecard and plots.
- Bring-your-own CSV with `--drop-cols` and ID-like column warnings.

## Planned

- Optional LightGBM / XGBoost baselines.
- Time-aware splitting for temporal data (`TimeSeriesSplit`).
- Optional hyperparameter search inside the Pipeline.
- Model calibration plots.
- Automatic coercion of numeric-looking text columns to numeric.
- Declaring numeric-coded columns as categorical (`--categorical-cols`).
- Basic date feature extraction.

## Known limitations

- Task detection is heuristic (override with `--task`).
- Numbers stored as text are treated as categorical; dates are not parsed.
- ID-column detection is a warning heuristic; columns to exclude must be passed through `--drop-cols`.
- The demo dataset is synthetic; speed targets assume under 100k rows on a standard laptop and depend on hardware
  and `DATAPREP_RF_ESTIMATORS`.
