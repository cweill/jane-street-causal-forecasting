import importlib
import inspect

import polars as pl
import pytest
import torch
from test_patrick_pipeline import panel, train

from src.artifacts import load_predictor
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource, RestrictedDateSource


def test_day_cache_is_bounded_cloned_and_preserves_date_restrictions():
    module = importlib.import_module("src.data.loader")
    assert hasattr(module, "CachedDaySource"), "bounded day cache missing"
    source = RestrictedDateSource(FrameSource(panel()), (0, 1, 2))
    reads = []
    original = source.day

    def counted(date):
        reads.append(date)
        return original(date)

    source.day = counted
    cached = module.CachedDaySource(source, capacity=2)
    first = cached.day(0)
    first.replace_column(
        first.get_column_index("weight"), pl.Series("weight", [0.0] * first.height)
    )
    assert cached.day(0).equals(original(0))
    cached.day(1)
    cached.day(2)
    cached.day(0)
    assert reads == [0, 1, 2, 0]
    with pytest.raises(ValueError, match="partition"):
        cached.day(3)


def test_fast_replay_matches_online_updates_and_rejects_future_label_influence(tmp_path):
    assert "prepartition_truth" in inspect.signature(APISimulator).parameters
    source, _, _ = train(tmp_path)
    cls = importlib.import_module("src.data.loader").CachedDaySource
    reference = load_predictor(tmp_path / "model")
    fast = load_predictor(tmp_path / "model")
    assert hasattr(fast, "fast_inference"), "streaming fast path switch missing"
    fast.fast_inference = True
    expected = APISimulator(source, [3, 4, 5, 6]).run(reference.predict)
    actual = APISimulator(cls(source), [3, 4, 5, 6], prepartition_truth=True).run(fast.predict)
    assert actual.predictions.equals(expected.predictions)
    assert actual.metric.score == expected.metric.score
    assert fast.update_log == reference.update_log
    for name, value in reference.models[0].state_dict().items():
        torch.testing.assert_close(fast.models[0].state_dict()[name], value, rtol=0, atol=0)
    poisoned = panel().with_columns(
        [
            pl.when(pl.col("date_id") >= 5)
            .then(999.0)
            .otherwise(pl.col(f"responder_{i}"))
            .alias(f"responder_{i}")
            for i in range(9)
        ]
    )
    clean, changed = load_predictor(tmp_path / "model"), load_predictor(tmp_path / "model")
    clean.fast_inference = changed.fast_inference = True
    a = APISimulator(cls(source), [3, 4, 5], prepartition_truth=True).run(clean.predict)
    b = APISimulator(cls(FrameSource(poisoned)), [3, 4, 5], prepartition_truth=True).run(
        changed.predict
    )
    assert a.predictions.equals(b.predictions)
    for name, value in clean.models[0].state_dict().items():
        torch.testing.assert_close(changed.models[0].state_dict()[name], value, rtol=0, atol=0)
