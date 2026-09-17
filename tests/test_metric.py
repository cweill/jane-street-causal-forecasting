import numpy as np
import pytest

from src.metric import WeightedZeroMeanR2, weighted_zero_mean_r2


def test_formula_is_zero_mean_and_not_centered():
    y, p, w = np.array([1.0, 3.0]), np.array([0.0, 2.0]), np.array([1.0, 2.0])
    assert weighted_zero_mean_r2(y, p, w) == pytest.approx(1 - 3 / 19)
    assert weighted_zero_mean_r2(y, y * 0, w) == 0
    assert weighted_zero_mean_r2(y, y, w) == 1
    assert weighted_zero_mean_r2(y, -y, w) == -3


def test_global_sums_not_mean_of_batch_scores_and_scored_mask():
    metric = WeightedZeroMeanR2()
    metric.update([1, 999], [0, 0], [1, 100], [True, False])
    metric.update([3], [3], [2])
    assert metric.score == pytest.approx(1 - 1 / 19)
    assert metric.rows == 2


@pytest.mark.parametrize(
    "y,p,w", [([1], [1, 2], [1]), ([1], [0], [-1]), ([1], [np.nan], [1]), ([np.inf], [0], [1])]
)
def test_invalid_metric_inputs_fail(y, p, w):
    with pytest.raises(ValueError):
        weighted_zero_mean_r2(y, p, w)


def test_degenerate_score_is_explicit_not_epsilon_dependent():
    with pytest.raises(ValueError, match="denominator"):
        weighted_zero_mean_r2([0], [0], [1])
    with pytest.raises(ValueError, match="denominator"):
        weighted_zero_mean_r2([1], [0], [0])
