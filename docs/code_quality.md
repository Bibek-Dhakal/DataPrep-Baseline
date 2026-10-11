# Code Quality

Hooks run Ruff and file checks on `git commit`, commit message validation on `commit-msg`, and tests on
`git push`. Manual fallbacks exist for every check.

## Environment Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Manual Execution Commands

```bash
pre-commit run --all-files          # all files
pre-commit run                      # staged files only
pre-commit run --files src/dataprep_baseline/cli.py
pre-commit run --from-ref origin/main --to-ref HEAD   # compare against a branch
```

## Isolated Tool Commands

```bash
ruff check .          # linter only
ruff check . --fix
ruff format .         # formatter only
pytest                # tests only
```

## Maintenance & Cache

```bash
pre-commit autoupdate
pre-commit clean
ruff clean
```

## Emergency Bypassing

```bash
git commit --no-verify
git push --no-verify
```

Use these only for genuine emergencies. CI and review still expect passing checks.
