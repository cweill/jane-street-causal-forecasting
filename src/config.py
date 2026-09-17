"""Strict configuration: every claimed improvement has an independent boolean switch."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from src.data.features import FeatureConfig
from src.models.grigoreva_gru import ModelConfig
from src.training.online import OnlineConfig


@dataclass(frozen=True)
class CVConfig:
    min_date: int = 700
    n_splits: int = 2
    validation_days: int = 200
    gap_days: int = 0
    min_train_days: int = 1
    max_train_days: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 0.0005
    device: str = "cpu"


@dataclass(frozen=True)
class EnsembleConfig:
    architectures: tuple[str, ...] = ("gru_mlp",)
    seed_ensembling: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baseline"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    def __post_init__(self):
        self.model.dimensions()
        if not self.name or self.training.epochs < 1 or self.training.learning_rate <= 0:
            raise ValueError("invalid experiment/training configuration")
        if self.training.device not in {"cpu", "cuda"}:
            raise ValueError("supported deterministic research devices: cpu, cuda")
        if not self.ensemble.seeds or len(set(self.ensemble.seeds)) != len(self.ensemble.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if not self.ensemble.architectures or len(set(self.ensemble.architectures)) != len(
            self.ensemble.architectures
        ):
            raise ValueError("architectures must be nonempty and unique")
        for architecture in self.ensemble.architectures:
            replace(self.model, architecture=architecture).dimensions()
        for value in (
            self.features.market_average,
            self.features.rolling,
            self.model.auxiliary_targets,
            self.online.enabled,
            self.ensemble.seed_ensembling,
        ):
            if type(value) is not bool:
                raise ValueError("ablation switches must be YAML booleans")

    @property
    def members(self):
        seeds = self.ensemble.seeds if self.ensemble.seed_ensembling else self.ensemble.seeds[:1]
        return [
            (replace(self.model, architecture=a), s)
            for a in self.ensemble.architectures
            for s in seeds
        ]

    def to_dict(self):
        return asdict(self)


def from_dict(raw):
    if raw.get("method") == "patrick":
        from src.patrick_config import from_dict as patrick_from_dict

        return patrick_from_dict(raw)
    raw = dict(raw)
    types = {
        "features": FeatureConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "online": OnlineConfig,
        "ensemble": EnsembleConfig,
        "cv": CVConfig,
    }
    for key, cls in types.items():
        if key in raw:
            args = dict(raw[key])
            for name in (
                "seeds",
                "architectures",
                "hidden_sizes",
                "linear_sizes",
                "dropout",
                "linear_dropout",
            ):
                if name in args and args[name] is not None:
                    args[name] = tuple(args[name])
            raw[key] = cls(**args)
    return ExperimentConfig(**raw)


def load_config(path: str | Path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a mapping")
    return from_dict(raw)
