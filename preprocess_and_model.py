"""Run the DataPrep-Baseline pipeline: ``python preprocess_and_model.py --help``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataprep_baseline.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
