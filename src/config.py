"""Patrick configuration loading and chronological split/ensemble settings."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CVConfig:
    min_date: int = 0
    n_splits: int = 1
    validation_days: int = 200
    gap_days: int = 120
    min_train_days: int = 1
    max_train_days: int | None = None
    train_end: int | None = None
    warmup_end: int | None = None
    validation_end: int | None = None


@dataclass(frozen=True)
class EnsembleConfig:
    architectures: tuple[str, ...] = ("patrick",)
    seed_ensembling: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)


def from_dict(raw):
    from src.patrick_config import from_dict as patrick_from_dict

    return patrick_from_dict(raw)


def load_config(path: str | Path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a mapping")
    return from_dict(raw)
