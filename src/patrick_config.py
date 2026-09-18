"""Patrick reconstruction configuration, separate from Grigoreva's defaults."""

from dataclasses import asdict, dataclass, field

from src.config import CVConfig, EnsembleConfig
from src.data.patrick_features import PatrickFeatureConfig
from src.models.patrick_yam import PatrickModelConfig
from src.training.patrick import PatrickOnlineConfig, PatrickTrainingConfig


@dataclass(frozen=True)
class PatrickInferenceConfig:
    stacked_ensemble: bool = False


@dataclass(frozen=True)
class PatrickConfig:
    method: str = "patrick"
    name: str = "patrick_reconstruction"
    features: PatrickFeatureConfig = field(default_factory=PatrickFeatureConfig)
    model: PatrickModelConfig = field(default_factory=PatrickModelConfig)
    training: PatrickTrainingConfig = field(default_factory=PatrickTrainingConfig)
    online: PatrickOnlineConfig = field(default_factory=PatrickOnlineConfig)
    inference: PatrickInferenceConfig = field(default_factory=PatrickInferenceConfig)
    ensemble: EnsembleConfig = field(
        default_factory=lambda: EnsembleConfig(architectures=("patrick",))
    )
    cv: CVConfig = field(default_factory=lambda: CVConfig(min_date=0, n_splits=1, gap_days=120))

    def __post_init__(self):
        if self.method != "patrick" or self.training.device not in {"cpu", "cuda"}:
            raise ValueError("invalid method/device")
        if (
            self.ensemble.architectures != ("patrick",)
            or not self.ensemble.seeds
            or len(set(self.ensemble.seeds)) != len(self.ensemble.seeds)
        ):
            raise ValueError("Patrick requires unique seeds and its own architecture")
        for n in (self.training.epochs, self.online.steps):
            if type(n) is not int or n < 1:
                raise ValueError("positive integer epochs/steps required")
        for config in (self.training, self.online):
            if (
                config.learning_rate <= 0
                or config.gradient_clip <= 0
                or len(config.betas) != 2
                or any(not 0 <= b < 1 for b in config.betas)
            ):
                raise ValueError("invalid optimizer settings")
        for flag in (
            self.model.post_norm,
            self.model.auxiliary_targets,
            self.model.balance_losses,
            self.training.recency_weighting,
            self.training.full_length_weighting,
            self.online.enabled,
            self.online.reset_daily_optimizer,
            self.ensemble.seed_ensembling,
            self.inference.stacked_ensemble,
        ):
            if type(flag) is not bool:
                raise ValueError("ablation switches must be booleans")

    @property
    def members(self):
        seeds = self.ensemble.seeds if self.ensemble.seed_ensembling else self.ensemble.seeds[:1]
        return [(self.model, s) for s in seeds]

    def to_dict(self):
        return asdict(self)


def from_dict(raw):
    raw = dict(raw)
    types = {
        "features": PatrickFeatureConfig,
        "model": PatrickModelConfig,
        "training": PatrickTrainingConfig,
        "online": PatrickOnlineConfig,
        "inference": PatrickInferenceConfig,
        "ensemble": EnsembleConfig,
        "cv": CVConfig,
    }
    for key, kind in types.items():
        if key in raw:
            values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw[key].items()}
            raw[key] = kind(**values)
    return PatrickConfig(**raw)
