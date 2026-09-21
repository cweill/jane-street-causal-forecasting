"""Train-only Patrick feature reconstruction; all assumptions are configuration values."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from src.data.normalization import TrainOnlyStandardizer
from src.data.schema import FEATURES, KEYS, TEST_COLUMNS, validate_test

BASE = [f for f in FEATURES if f not in {"feature_09", "feature_10", "feature_11"}]


@dataclass(frozen=True)
class PatrickFeatureConfig:
    category_columns: tuple[str, ...] = ("feature_09", "feature_10", "feature_11")
    category_capacities: tuple[int, ...] = (83, 13, 540)
    time_steps: int = 968
    time_epsilon: float = 0.0001

    def __post_init__(self):
        if tuple(self.category_columns) != ("feature_09", "feature_10", "feature_11"):
            raise ValueError("this reconstruction maps the three omitted features to categories")
        if len(self.category_capacities) != 3 or min(self.category_capacities) < 1:
            raise ValueError("three positive categorical capacities required")
        if self.time_steps < 2 or not 0 < self.time_epsilon < 0.5:
            raise ValueError("invalid Gaussian time transform")


class PatrickFeatures:
    def __init__(self, config, training_dates):
        self.config = config
        self.names = [*BASE, "gaussian_time"]
        self.scaler = TrainOnlyStandardizer(training_dates, len(BASE))
        self.vocabularies = [set() for _ in range(3)]
        self.maps = None
        normal = NormalDist()
        self.time_table = np.array(
            [
                normal.inv_cdf(
                    np.clip(
                        (t + 0.5) / config.time_steps, config.time_epsilon, 1 - config.time_epsilon
                    )
                )
                for t in range(config.time_steps)
            ],
            dtype=np.float32,
        )

    @property
    def vocab_sizes(self):
        # Reserve a separate zero embedding for unknown/missing values.
        return tuple(n + 1 for n in self.config.category_capacities)

    def update(self, day, date):
        if self.maps is not None or date not in self.scaler.training_dates:
            raise ValueError("preprocessing may only fit declared training dates before freeze")
        if day["date_id"].unique().to_list() != [date]:
            raise ValueError("training date mismatch")
        self.scaler.update(day.select(BASE).to_numpy(), date=date)
        for values, name in zip(self.vocabularies, self.config.category_columns, strict=True):
            values.update(float(v) for v in day[name].drop_nulls().unique() if np.isfinite(v))

    def freeze(self):
        if any(
            len(v) > n
            for v, n in zip(self.vocabularies, self.config.category_capacities, strict=True)
        ):
            raise ValueError("observed categories exceed configured capacity")
        self.maps = [
            {v: i + 1 for i, v in enumerate(sorted(values))} for values in self.vocabularies
        ]
        self.scaler.freeze()
        return self

    def transform(self, public):
        validate_test(public)
        return self._transform_rows(public)

    def transform_day(self, public):
        """Vectorized row-local transform; never fits statistics from this day."""
        if set(public.columns) != set(TEST_COLUMNS):
            raise ValueError("only API-visible columns are allowed; responder leak")
        if (
            public.is_empty()
            or public["date_id"].n_unique() != 1
            or public.select(KEYS).is_duplicated().any()
        ):
            raise ValueError("expected one nonempty day with unique keys")
        return self._transform_rows(public)

    def _transform_rows(self, public):
        if self.maps is None:
            raise ValueError("freeze preprocessing before use")
        x = self.scaler.transform(public.select(BASE).to_numpy())
        times = public["time_id"].to_numpy()
        t = self.time_table[np.clip(times, 0, len(self.time_table) - 1)]
        numeric = np.concatenate([x, t[:, None]], axis=1).astype(np.float32)
        categorical = np.array(
            [
                [
                    mapping.get(v, 0) if v is not None else 0
                    for mapping, v in zip(self.maps, row, strict=True)
                ]
                for row in public.select(self.config.category_columns).iter_rows()
            ],
            dtype=np.int64,
        )
        return numeric, categorical

    def state_dict(self):
        return {
            "scaler": self.scaler.state_dict(),
            "vocabularies": [sorted(values) for values in self.vocabularies],
        }

    def load_state_dict(self, state):
        self.scaler = TrainOnlyStandardizer.from_state_dict(state["scaler"])
        self.vocabularies = [set(values) for values in state["vocabularies"]]
        self.freeze()

    def reset_clock(self):
        # No feature clock or responder state: fixed time transform, frozen vocabularies.
        pass
