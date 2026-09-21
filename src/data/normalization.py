"""Frozen training-only numerical statistics for Patrick preprocessing."""

import numpy as np


class TrainOnlyStandardizer:
    """Mergeable observed-value moments; zero imputation before standardization.

    All-missing and constant columns get a unit scale; nonfinite outputs become zero.
    The allowed dates are fixed at construction and the scaler freezes before replay.
    """

    def __init__(self, training_dates, size):
        self.training_dates = frozenset(training_dates)
        self.count = np.zeros(size, dtype=np.int64)
        self.mean = np.zeros(size)
        self.m2 = np.zeros(size)
        self.scale = np.ones(size)
        self.frozen = False

    def update(self, x, *, date):
        if self.frozen:
            raise ValueError("standardizer is frozen")
        if date not in self.training_dates:
            raise ValueError("standardizer may only fit declared training dates")
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        count = finite.sum(axis=0)
        mean = np.divide(
            np.where(finite, x, 0.0).sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0
        )
        m2 = np.where(finite, x - mean, 0.0) ** 2
        total = self.count + count
        delta = mean - self.mean
        self.mean += delta * count / np.maximum(total, 1)
        self.m2 += m2.sum(axis=0) + delta**2 * self.count * count / np.maximum(total, 1)
        self.count = total

    def freeze(self):
        self.scale = np.sqrt(self.m2 / np.maximum(self.count - 1, 1))
        self.scale[(self.scale < 1e-12) | ~np.isfinite(self.scale)] = 1.0
        self.frozen = True
        return self

    def transform(self, x):
        if not self.frozen:
            raise ValueError("freeze training standardizer before transformation")
        x = np.where(np.isfinite(x), x, 0.0)
        return np.nan_to_num((x - self.mean) / self.scale).astype(np.float32)

    def state_dict(self):
        return {
            "training_dates": sorted(self.training_dates),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "m2": self.m2.tolist(),
            "time_index": None,
            "time_bounds": None,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("time_index") is not None or state.get("time_bounds") is not None:
            raise ValueError("Patrick normalizer does not scale or adapt time bounds")
        scaler = cls(state["training_dates"], len(state["mean"]))
        for name in ("mean", "scale", "count", "m2"):
            setattr(scaler, name, np.array(state[name]))
        scaler.frozen = True
        return scaler
