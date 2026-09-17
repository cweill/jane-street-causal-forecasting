import numpy as np
import polars as pl
import pytest

from src.cv import temporal_folds
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.data.schema import RESPONDERS


def lag_predictor():
    last = 0.0

    def predict(test, lags):
        nonlocal last
        if lags is not None and not lags.is_empty():
            last = lags["responder_6_lag_1"].mean()
        return test.select("row_id").with_columns(pl.lit(last).alias("responder_6"))

    return predict


def test_mutating_all_unreleased_responders_cannot_change_predictions(panel):
    changed = panel.with_columns(
        [
            pl.when(pl.col("date_id") >= 3).then(pl.col(c) * -1000).otherwise(pl.col(c)).alias(c)
            for c in RESPONDERS
        ]
    )
    a = APISimulator(FrameSource(panel), [2, 3, 4]).run(lag_predictor()).predictions
    b = APISimulator(FrameSource(changed), [2, 3, 4]).run(lag_predictor()).predictions
    n = panel.filter(pl.col("date_id").is_in([2, 3])).height
    np.testing.assert_array_equal(a["responder_6"][:n], b["responder_6"][:n])
    assert a["responder_6"][-1] != b["responder_6"][-1]  # release is effective next day


def test_temporal_splits_are_deterministic_whole_days_and_gap_is_replayed():
    dates = list(range(700, 1699))
    folds = temporal_folds(dates, n_splits=2, validation_days=200)
    assert folds == temporal_folds(dates[::-1] + dates[:10], n_splits=2, validation_days=200)
    assert (folds[0].train_dates[-1], folds[0].validation_dates[0]) == (1298, 1299)
    assert (folds[1].train_dates[-1], folds[1].validation_dates[0]) == (1498, 1499)
    gap = temporal_folds(dates, n_splits=1, validation_days=200, gap_days=200)[0]
    assert gap.train_dates == folds[0].train_dates
    assert gap.warmup_dates == tuple(range(1299, 1499))
    assert gap.validation_dates == folds[1].validation_dates
    assert max(gap.train_dates) < min(gap.warmup_dates) < min(gap.validation_dates)


def test_insufficient_dates_fail_instead_of_silently_resizing_folds():
    with pytest.raises(ValueError):
        temporal_folds([0, 1], n_splits=2, validation_days=200)
