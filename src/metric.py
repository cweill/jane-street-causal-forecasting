"""Competition score: 1 - sum(w * (y - p)^2) / sum(w * y^2).

Accumulate across all scored rows, never average per-day R². A zero denominator
is undefined; raise rather than introduce an undocumented epsilon into scoring.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightedZeroMeanR2:
    sse: float = 0.0
    denominator: float = 0.0
    rows: int = 0

    def update(self, y, prediction, weight, is_scored=None):
        y, prediction, weight = (np.asarray(a, dtype=np.float64) for a in (y, prediction, weight))
        if y.ndim != 1 or y.shape != prediction.shape or y.shape != weight.shape:
            raise ValueError("metric inputs must have identical one-dimensional shapes")
        if not all(np.isfinite(a).all() for a in (y, prediction, weight)):
            raise ValueError("metric inputs must be finite")
        if (weight < 0).any():
            raise ValueError("weights must be nonnegative")
        mask = np.ones(y.shape, dtype=bool) if is_scored is None else np.asarray(is_scored)
        if mask.dtype != np.bool_ or mask.shape != y.shape:
            raise ValueError("is_scored must be a boolean vector matching the rows")
        y, prediction, weight = y[mask], prediction[mask], weight[mask]
        self.sse += float(np.sum(weight * np.square(y - prediction)))
        self.denominator += float(np.sum(weight * np.square(y)))
        self.rows += len(y)
        if not np.isfinite([self.sse, self.denominator]).all():
            raise ValueError("metric accumulation overflow")
        return self

    @property
    def score(self):
        if self.denominator <= 0:
            raise ValueError("zero denominator: score is undefined")
        return 1.0 - self.sse / self.denominator


def weighted_zero_mean_r2(y, prediction, weight, is_scored=None):
    return WeightedZeroMeanR2().update(y, prediction, weight, is_scored).score
