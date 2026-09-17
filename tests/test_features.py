import numpy as np
import polars as pl
import pytest

from src.data.features import SELECTED, FeatureConfig, FeaturePipeline, TrainOnlyStandardizer
from src.data.schema import test_view as api_view


def test_market_and_rolling_match_hand_computed_causal_values(panel):
    pipeline = FeaturePipeline(FeatureConfig(market_average=True, rolling=True, rolling_window=3))
    day = api_view(panel.filter(pl.col("date_id") == 0))
    frames = day.partition_by("time_id", maintain_order=True)
    results = [pipeline.transform(f) for f in frames]
    f = SELECTED[0]
    raw = day.filter(pl.col("symbol_id") == 2)[f].to_numpy()
    diff = pipeline.names.index(f + "_diff_rolling_avg_3")
    std = pipeline.names.index(f + "_rolling_std_3")
    avg = pipeline.names.index(f + "_avg_per_date_time")
    assert np.isnan(results[0][0, diff])
    assert results[2][0, diff] == pytest.approx(raw[2] - raw[:3].mean())
    assert results[2][0, std] == pytest.approx(raw[:3].std(ddof=1))
    assert results[3][0, diff] == pytest.approx(raw[3] - raw[1:4].mean())
    assert results[0][0, avg] == pytest.approx(frames[0][f].mean())
    assert len(pipeline.names) == 125
    assert "feature_09" not in pipeline.names


def test_features_do_not_accept_any_responder_or_future_column(panel):
    pipeline = FeaturePipeline(FeatureConfig())
    batch = api_view(panel.head(2))
    for col in ("responder_6", "responder_6_lag_1", "future_target"):
        with pytest.raises(ValueError, match="schema"):
            pipeline.transform(batch.with_columns(pl.lit(1).alias(col)))


def test_out_of_order_features_rejected_and_rolling_continues_across_days(panel):
    p = FeaturePipeline(FeatureConfig(rolling=True, rolling_window=3))
    day0 = api_view(panel.filter(pl.col("date_id") == 0))
    for batch in day0.partition_by("time_id", maintain_order=True):
        p.transform(batch)
    with pytest.raises(ValueError, match="chronological"):
        p.transform(day0.head(2))
    out = p.transform(api_view(panel.filter(pl.col("date_id") == 1).head(2)))
    assert np.isfinite(out).all()


def test_scaler_rejects_validation_fit_and_freezes(panel):
    scaler = TrainOnlyStandardizer((0, 1), 2)
    scaler.update(np.array([[1.0, np.nan], [3.0, np.nan]]), date=0)
    with pytest.raises(ValueError, match="training"):
        scaler.update(np.ones((2, 2)), date=2)
    scaler.freeze()
    np.testing.assert_allclose(scaler.transform(np.array([[2.0, np.nan]])), [[0.0, 0.0]])
    with pytest.raises(ValueError, match="frozen"):
        scaler.update(np.ones((2, 2)), date=1)


def test_future_feature_perturbation_preserves_prefix(panel):
    a = FeaturePipeline(FeatureConfig(market_average=True, rolling=True, rolling_window=3))
    b = FeaturePipeline(a.config)
    for batch in api_view(panel).partition_by(["date_id", "time_id"], maintain_order=True):
        changed = batch
        if batch["date_id"][0] >= 3:
            changed = batch.with_columns(pl.col("feature_06") + 1e6)
        x, z = a.transform(batch), b.transform(changed)
        if batch["date_id"][0] < 3:
            np.testing.assert_array_equal(x, z)


def test_time_feature_clips_to_training_range_until_released_day_extends_it():
    scaler = TrainOnlyStandardizer((0,), 2, time_index=1)
    scaler.update(np.array([[0.0, 0.0], [0.0, 2.0]]), date=0)
    scaler.freeze()
    assert scaler.transform(np.array([[0.0, 100.0]]))[0, 1] == pytest.approx(1 / np.sqrt(2))
    old_mean, old_scale = scaler.mean.copy(), scaler.scale.copy()
    scaler.observe_released_times(np.array([0, 4]))
    assert scaler.transform(np.array([[0.0, 100.0]]))[0, 1] == pytest.approx(3 / np.sqrt(2))
    np.testing.assert_array_equal(scaler.mean, old_mean)
    np.testing.assert_array_equal(scaler.scale, old_scale)
