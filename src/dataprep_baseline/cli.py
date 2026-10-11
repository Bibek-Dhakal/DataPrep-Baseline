"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

from .config import ENCODERS, SCALERS, TASKS, Config
from .data import load_dataset
from .evaluate import primary_metric
from .pipeline import export_artifacts, run_experiment

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataprep-baseline",
        description="Leak-free preprocessing and baseline modeling. Runs a demo dataset without --data.",
    )
    p.add_argument("--data", dest="data_path", type=Path, help="Path to input CSV file")
    p.add_argument("--target", help="Target column name (required with --data)")
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--seed", dest="random_state", type=int)
    p.add_argument("--test-size", type=float)
    p.add_argument("--cv-folds", type=int)
    p.add_argument("--scaler", choices=SCALERS)
    p.add_argument("--encoder", dest="categorical_encoder", choices=ENCODERS)
    p.add_argument("--output-dir", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.perf_counter()
    args = vars(build_parser().parse_args(argv))
    overrides = {k: v for k, v in args.items() if v is not None}

    try:
        cfg = replace(Config.from_env(), **overrides)
        X, y = load_dataset(cfg.data_path, cfg.target, cfg.random_state)
        result = run_experiment(X, y, cfg)
    except ValueError as exc:
        logger.error("Error: %s", exc)
        return 2

    paths = export_artifacts(result, cfg)
    print("\nScorecard (cv_* = cross-validation on train split, test_* = hold-out):")
    print(result.scorecard.round(4).to_string())
    print(f"\nBest model (by CV {primary_metric(result.task)}): {result.best_name}")
    print("Artifacts: " + ", ".join(str(p) for p in paths))
    logger.info("Total runtime: %.2fs", time.perf_counter() - start)
    return 0
