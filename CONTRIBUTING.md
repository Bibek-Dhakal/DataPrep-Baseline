# Contributing

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Code style and tests

- Formatting and linting: Ruff (see [docs/code_quality.md](docs/code_quality.md)).
- Tests: `pytest` (see [docs/testing.md](docs/testing.md)).
- Keep constants and configuration in `src/dataprep_baseline/config.py` only.
- Core invariant: transformers are fit on training data only, always inside a `Pipeline`.

## Commit messages (Conventional Commits, enforced)

All commit messages and PR titles must follow `type(scope): description`. They drive automated versioning and
`CHANGELOG.md` generation through `release-please` Release PRs.

| Prefix                              | Effect                               |
|-------------------------------------|--------------------------------------|
| `feat:`                             | Minor version bump                   |
| `fix:`                              | Patch version bump                   |
| `feat!:` / `BREAKING CHANGE:`       | Major version bump                   |
| `docs:`, `chore:`, `style:`, `test:`, `refactor:`, `perf:`, `build:`, `ci:` | No release on their own |

## Pull requests

1. Branch from `main` and keep changes focused.
2. Make sure `pytest` and `ruff check .` pass.
3. Use a Conventional Commit style PR title.
4. Merging to `main` updates the Release PR; merging the Release PR publishes the release.
