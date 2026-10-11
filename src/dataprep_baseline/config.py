"""Central configuration. Defaults live here only; env vars and CLI flags override them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ENV_PREFIX = "DATAPREP_"
TASKS = ("auto", "classification", "regression")
SCALERS = ("robust", "standard")
ENCODERS = ("onehot", "target")


def _env(name: str, default: str) -> str:
    return os.getenv(f"{ENV_PREFIX}{name}", default)


@dataclass(frozen=True)
class Config:
    data_path: Path | None = None
    target: str | None = None
    task: str = "auto"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    scaler: str = "robust"
    categorical_encoder: str = "onehot"
    n_jobs: int = -1
    rf_estimators: int = 100
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.scaler not in SCALERS:
            raise ValueError(f"scaler must be one of {SCALERS}, got {self.scaler!r}")
        if self.categorical_encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {self.categorical_encoder!r}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1 (exclusive)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.rf_estimators < 1:
            raise ValueError("rf_estimators must be at least 1")

    @classmethod
    def from_env(cls) -> Config:
        data_path = _env("DATA_PATH", "")
        return cls(
            data_path=Path(data_path) if data_path else None,
            target=_env("TARGET", "") or None,
            task=_env("TASK", cls.task),
            random_state=int(_env("RANDOM_STATE", str(cls.random_state))),
            test_size=float(_env("TEST_SIZE", str(cls.test_size))),
            cv_folds=int(_env("CV_FOLDS", str(cls.cv_folds))),
            scaler=_env("SCALER", cls.scaler),
            categorical_encoder=_env("ENCODER", cls.categorical_encoder),
            n_jobs=int(_env("N_JOBS", str(cls.n_jobs))),
            rf_estimators=int(_env("RF_ESTIMATORS", str(cls.rf_estimators))),
            output_dir=Path(_env("OUTPUT_DIR", str(cls.output_dir))),
        )
