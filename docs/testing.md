# Testing

```bash
pytest                      # full suite with coverage
pytest -k leakage           # leakage-focused tests
pytest -k drop_cols         # dataset column handling
```

| Tier        | Coverage                                                                            |
|-------------|-------------------------------------------------------------------------------------|
| Unit        | Leakage audit, unseen categories at inference, regression task, CLI validation, `--drop-cols` parsing and validation, ID-like column warning, missing file error. |
| Invariant   | A spy transformer proves `fit` never sees hold-out rows and `predict` never refits. |
| Integration | Reproducibility across runs; CLI demo run exports all artifacts; CLI run on a custom CSV with `--drop-cols`. |

Tests use small synthetic datasets (200-300 rows) and few trees for speed. No external data, downloads or mocks are
needed. `pytest` also runs on pre-push through pre-commit.

Real-world datasets (see [data/README.md](data/README.md)) are used for manual verification only and are not part of
the automated suite.
