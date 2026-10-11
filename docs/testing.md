# Testing

```bash
pytest                      # full suite with coverage
pytest -k leakage           # leakage-focused tests
```

| Tier        | Coverage                                                                            |
|-------------|-------------------------------------------------------------------------------------|
| Unit        | Leakage audit, unseen categories at inference, regression task, CLI validation.    |
| Invariant   | A spy transformer proves `fit` never sees hold-out rows and `predict` never refits. |
| Integration | Reproducibility across runs; CLI demo run exports all artifacts.                    |

Tests use small synthetic datasets (300 rows) and few trees for speed. No external data or mocks are needed.
`pytest` also runs on pre-push through pre-commit.
