"""Published feature set, computed from strictly chronological API-visible batches."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.data.schema import FEATURES, validate_test

# Fixed by the author's public code; never reselected using validation responders.
SELECTED = [
    f"feature_{i:02d}" for i in (6, 4, 7, 36, 60, 45, 56, 5, 51, 19, 66, 59, 54, 70, 71, 72)
]
BASE = [f for f in FEATURES if f not in {"feature_09", "feature_10", "feature_11"}]


@dataclass(frozen=True)
class FeatureConfig:
    market_average: bool = False
    rolling: bool = False
    rolling_window: int = 1000

    def __post_init__(self):
        if self.rolling_window < 2:
            raise ValueError("rolling_window must be at least 2")


class FeaturePipeline:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.names = list(BASE)
        if config.rolling:
            self.names += [f"{f}_diff_rolling_avg_{config.rolling_window}" for f in SELECTED]
            self.names += [f"{f}_rolling_std_{config.rolling_window}" for f in SELECTED]
        if config.market_average:
            self.names += [f"{f}_avg_per_date_time" for f in SELECTED]
        self.names += ["feature_time_id"]
        self._last_key = None
        self._history = {}
        self._moments = {}

    def transform(self, test):
        validate_test(test)
        key = (int(test["date_id"][0]), int(test["time_id"][0]))
        if self._last_key is not None and key <= self._last_key:
            raise ValueError("features require strictly chronological batches")
        self._last_key = key
        raw = test.select(BASE).to_numpy().astype(np.float64)
        raw[~np.isfinite(raw)] = np.nan
        selected = raw[:, [BASE.index(f) for f in SELECTED]]
        parts = [raw]
        if self.config.rolling:
            means, stds = [], []
            for symbol, values in zip(test["symbol_id"], selected, strict=True):
                history = self._history.setdefault(symbol, deque())
                total, squares, counts = self._moments.setdefault(
                    symbol, (np.zeros(16), np.zeros(16), np.zeros(16, dtype=np.int64))
                )
                if len(history) == self.config.rolling_window:
                    old = history.popleft()
                    finite = np.isfinite(old)
                    safe = np.where(finite, old, 0.0)
                    total -= safe
                    squares -= safe**2
                    counts -= finite
                history.append(values.copy())
                finite = np.isfinite(values)
                safe = np.where(finite, values, 0.0)
                total += safe
                squares += safe**2
                counts += finite
                # Match the training rolling operation: full valid window, ddof=1,
                # current observation included, no forward fill or future backfill.
                valid = counts == self.config.rolling_window
                mean = total / np.maximum(counts, 1)
                variance = np.maximum(squares - total * mean, 0.0) / np.maximum(counts - 1, 1)
                means.append(np.where(valid, mean, np.nan))
                stds.append(np.where(valid, np.sqrt(variance), np.nan))
            parts.extend([selected - np.array(means), np.array(stds)])
        if self.config.market_average:
            count = np.isfinite(selected).sum(axis=0)
            mean = np.divide(
                np.nansum(selected, axis=0), count, out=np.full(16, np.nan), where=count > 0
            )
            parts.append(np.broadcast_to(mean, selected.shape))
        parts.append(np.full((test.height, 1), key[1]))
        return np.concatenate(parts, axis=1)

    def state_dict(self):
        return {
            "last_key": self._last_key,
            "history": {str(k): np.array(v).tolist() for k, v in self._history.items()},
        }

    def reset_clock(self):
        """Begin an API stream whose date IDs restart at zero; retain rolling history."""
        self._last_key = None

    def load_state_dict(self, state):
        self._last_key = None if state["last_key"] is None else tuple(state["last_key"])
        self._history, self._moments = {}, {}
        for symbol, values in state["history"].items():
            a = np.array(values, dtype=np.float64)
            self._history[int(symbol)] = deque(a)
            self._moments[int(symbol)] = (
                np.nansum(a, axis=0),
                np.nansum(a * a, axis=0),
                np.isfinite(a).sum(axis=0),
            )


class TrainOnlyStandardizer:
    """Mergeable observed-value moments; zero imputation before standardization.

    All-missing and constant columns get a unit scale; nonfinite outputs become zero.
    The allowed dates are fixed at construction and the scaler freezes before replay.
    """

    def __init__(self, training_dates, size, time_index=None):
        self.training_dates = frozenset(training_dates)
        self.count = np.zeros(size, dtype=np.int64)
        self.mean = np.zeros(size)
        self.m2 = np.zeros(size)
        self.scale = np.ones(size)
        self.frozen = False
        self.time_index = time_index
        self.time_bounds = None

    def update(self, x, *, date):
        if self.frozen:
            raise ValueError("standardizer is frozen")
        if date not in self.training_dates:
            raise ValueError("standardizer may only fit declared training dates")
        x = np.asarray(x, dtype=np.float64)
        if self.time_index is not None:
            self.observe_released_times(x[:, self.time_index])
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
        if self.time_index is not None and self.time_bounds is not None:
            x[:, self.time_index] = np.clip(x[:, self.time_index], *self.time_bounds)
        return np.nan_to_num((x - self.mean) / self.scale).astype(np.float32)

    def observe_released_times(self, times):
        """Extend time clipping bounds from training or a newly released cached day.

        This never changes fitted feature means or standard deviations.
        """
        times = np.asarray(times)
        times = times[np.isfinite(times)]
        if not len(times):
            return
        low, high = float(times.min()), float(times.max())
        if self.time_bounds is not None:
            low, high = min(low, self.time_bounds[0]), max(high, self.time_bounds[1])
        self.time_bounds = (low, high)

    def state_dict(self):
        return {
            "training_dates": sorted(self.training_dates),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "m2": self.m2.tolist(),
            "time_index": self.time_index,
            "time_bounds": self.time_bounds,
        }

    @classmethod
    def from_state_dict(cls, state):
        scaler = cls(state["training_dates"], len(state["mean"]), state["time_index"])
        scaler.time_bounds = state["time_bounds"]
        for name in ("mean", "scale", "count", "m2"):
            setattr(scaler, name, np.array(state[name]))
        scaler.frozen = True
        return scaler
