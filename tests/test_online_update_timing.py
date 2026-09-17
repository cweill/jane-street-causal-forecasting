import copy

import numpy as np
import polars as pl
import pytest
import torch

from src.data.api_simulator import APISimulator, previous_day_lags
from src.data.features import FeatureConfig, FeaturePipeline, TrainOnlyStandardizer
from src.data.loader import FrameSource
from src.data.schema import RESPONDERS
from src.data.schema import test_view as api_view
from src.models.grigoreva_gru import GrigorevaGRU, ModelConfig
from src.training.online import OnlineConfig, StreamingPredictor


def make_predictor(online=True, seeds=(0,), auxiliary=True):
    features = FeaturePipeline(FeatureConfig(market_average=True, rolling=True, rolling_window=3))
    scaler = TrainOnlyStandardizer((0,), len(features.names))
    scaler.update(np.zeros((2, len(features.names))), date=0)
    scaler.freeze()
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        models.append(
            GrigorevaGRU(
                len(features.names),
                ModelConfig(
                    auxiliary_targets=auxiliary,
                    hidden_sizes=(4,),
                    linear_sizes=(),
                    dropout=(0.1,),
                    linear_dropout=(),
                ),
            )
        )
    return StreamingPredictor(models, features, scaler, OnlineConfig(enabled=online), seeds=seeds)


def test_updates_only_at_next_time_zero_using_previous_day_including_unscored(panel):
    predictor = make_predictor()
    initial = copy.deepcopy(predictor.models[0].state_dict())
    APISimulator(FrameSource(panel), [2, 3, 4]).run(predictor.predict)
    assert [(e["released_at"], e["source_date"], e["rows"]) for e in predictor.update_log] == [
        ([3, 0], 2, 10),
        ([4, 0], 3, 10),
    ]
    assert any(not torch.equal(initial[k], v) for k, v in predictor.models[0].state_dict().items())
    assert predictor.cached_dates == {4}  # final day never trains without its next release


@pytest.mark.parametrize("auxiliary", [True, False])
def test_all_future_responders_cannot_affect_model_or_predictions_before_release(panel, auxiliary):
    a, b = make_predictor(auxiliary=auxiliary), make_predictor(auxiliary=auxiliary)
    changed = panel.with_columns(
        [
            pl.when(pl.col("date_id") >= 3).then(pl.col(c) * -10).otherwise(pl.col(c)).alias(c)
            for c in RESPONDERS
        ]
    )
    pa = APISimulator(FrameSource(panel), [2, 3]).run(a.predict).predictions
    pb = APISimulator(FrameSource(changed), [2, 3]).run(b.predict).predictions
    np.testing.assert_array_equal(pa["responder_6"], pb["responder_6"])
    for name, value in a.models[0].state_dict().items():
        torch.testing.assert_close(value, b.models[0].state_dict()[name], rtol=0, atol=0)


def test_auxiliary_responders_never_drive_online_loss(panel):
    changed = panel.with_columns(
        [(-pl.col(c) * 100).alias(c) for c in RESPONDERS if c != "responder_6"]
    )
    a, b = make_predictor(), make_predictor()
    pa = APISimulator(FrameSource(panel), [2, 3, 4]).run(a.predict).predictions
    pb = APISimulator(FrameSource(changed), [2, 3, 4]).run(b.predict).predictions
    np.testing.assert_array_equal(pa["responder_6"], pb["responder_6"])


def test_disabled_online_never_changes_weights(panel):
    predictor = make_predictor(online=False)
    before = copy.deepcopy(predictor.models[0].state_dict())
    APISimulator(FrameSource(panel), [2, 3, 4]).run(predictor.predict)
    assert predictor.update_log == []
    for key, value in predictor.models[0].state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_lag_date_mismatch_or_midday_release_fails(panel):
    for date_shift, time in [(1, 0), (0, 1)]:
        predictor = make_predictor()
        batch = api_view(panel.filter((pl.col("date_id") == 3) & (pl.col("time_id") == time)))
        lags = (
            previous_day_lags(panel.filter(pl.col("date_id") == 2), 3 + date_shift)
            if not date_shift
            else (
                previous_day_lags(panel.filter(pl.col("date_id") == 2), 3).with_columns(
                    pl.lit(4).alias("date_id")
                )
            )
        )
        with pytest.raises(ValueError, match="lags"):
            predictor.predict(batch, lags)


def test_missing_lag_labels_do_not_drop_recurrent_steps_or_invent_labels(panel):
    predictor = make_predictor()
    source = FrameSource(panel)

    def wrapped(test, lags):
        if lags is not None:
            lags = lags.filter((pl.col("time_id") != 2) & (pl.col("symbol_id") == 9))
        return predictor.predict(test, lags)

    APISimulator(source, [2, 3]).run(wrapped)
    assert predictor.update_log[0]["rows"] == 4


def test_seed_ensemble_is_average_and_members_are_independent(panel):
    a, b, ensemble = (
        make_predictor(seeds=(0,)),
        make_predictor(seeds=(1,)),
        make_predictor(seeds=(0, 1)),
    )
    predictions = [
        APISimulator(FrameSource(panel), [2, 3])
        .run(p.predict)
        .predictions["responder_6"]
        .to_numpy()
        for p in (a, b, ensemble)
    ]
    np.testing.assert_allclose(
        predictions[2], (predictions[0] + predictions[1]) / 2, rtol=1e-6, atol=1e-6
    )
    assert ensemble.models[0] is not ensemble.models[1]
    assert not np.array_equal(predictions[0], predictions[1])


def test_missing_symbols_row_permutation_and_day_reset(panel):
    a, b = make_predictor(online=False), make_predictor(online=False)
    source = FrameSource(panel.filter(~((pl.col("time_id") == 1) & (pl.col("symbol_id") == 9))))
    pa = APISimulator(source, [2, 3]).run(a.predict).predictions

    def permute(test, lags):
        return b.predict(test.reverse(), lags).reverse()

    pb = APISimulator(source, [2, 3]).run(permute).predictions
    np.testing.assert_allclose(pa["responder_6"], pb["responder_6"], atol=1e-7)
    assert a.hidden_dates == {3}


def test_new_day_predictions_match_fresh_recurrent_state(panel):
    a = make_predictor(online=False)
    APISimulator(FrameSource(panel), [2]).run(a.predict)
    b = StreamingPredictor(
        [copy.deepcopy(a.models[0])],
        copy.deepcopy(a.features),
        copy.deepcopy(a.scaler),
        OnlineConfig(False),
        seeds=[0],
    )
    pa = APISimulator(FrameSource(panel), [3]).run(a.predict).predictions
    pb = APISimulator(FrameSource(panel), [3]).run(b.predict).predictions
    np.testing.assert_array_equal(pa["responder_6"], pb["responder_6"])
