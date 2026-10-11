# Roadmap

## Current (v0.1)

- Leak-free Pipeline with classification and regression baselines, audit, scorecard and plots.

## Planned

- Optional LightGBM / XGBoost baselines.
- Time-aware splitting for temporal data (`TimeSeriesSplit`).
- Optional hyperparameter search inside the Pipeline.
- Model calibration plots.

## Known limitations

- Task detection is heuristic (override with `--task`).
- The demo dataset is synthetic; speed targets assume under 100k rows on a standard laptop.
