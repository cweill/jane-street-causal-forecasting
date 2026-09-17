import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import torch

from src.artifacts import load_predictor, save_predictor
from src.config import from_dict
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.data.schema import test_view as public_view
from src.data.synthetic import synthetic_panel


def api():
    assert importlib.util.find_spec("src.training.patrick"), "Patrick training not implemented"
    return importlib.import_module("src.training.patrick")


def panel():
    return synthetic_panel(days=7, times=5, symbols=2).with_columns(
        [(pl.col("symbol_id") % 2).cast(pl.Float32).alias(f"feature_{i:02d}") for i in [9, 10, 11]]
    )


def tiny_config():
    return from_dict(
        {
            "method": "patrick",
            "model": {
                "d_model": 8,
                "nheads": 2,
                "d_hidden": 16,
                "layers": 2,
                "rnn_multiplier": 2,
                "head_sizes": [12, 8],
            },
            "training": {"epochs": 1},
            "online": {"enabled": True},
        }
    )


def train(tmp_path):
    module, config = api(), tiny_config()
    source = FrameSource(panel())
    prepared = module.prepare_training(source, (0, 1, 2), config.features, tmp_path / "train")
    model, _ = module.train_model(prepared, config.model, training=config.training, seed=0)
    save_predictor(
        tmp_path / "model", [model], prepared.features, prepared.scaler, config.online, [0]
    )
    return source, prepared, config


def test_preprocessing_fits_only_training_and_unknown_categories_stay_unknown(tmp_path):
    module, config = api(), tiny_config()
    source = FrameSource(panel())

    class TrainingOnly:
        def dates(self):
            return source.dates()

        def day(self, date):
            assert date in (0, 1, 2), "future feature information read during fit"
            return source.day(date)

    prepared = module.prepare_training(
        TrainingOnly(), (0, 1, 2), config.features, tmp_path / "train"
    )
    before = prepared.features.state_dict()
    test = public_view(source.day(3)).filter(pl.col("time_id") == 0)
    test = test.with_columns(pl.lit(999).alias("feature_09"))
    x, categories = prepared.features.transform(test)
    assert x.shape[1] == 77 and np.isfinite(x).all()
    assert np.all(categories[:, 0] == 0)
    assert before == prepared.features.state_dict()
    with pytest.raises(ValueError, match="responder"):
        prepared.features.transform(test.with_columns(pl.lit(99).alias("responder_6")))


def test_stream_checkpoint_online_timing_and_future_responders(tmp_path):
    source, prepared, _ = train(tmp_path)
    a, b = load_predictor(tmp_path / "model"), load_predictor(tmp_path / "model")
    altered = panel().with_columns(
        [
            pl.when(pl.col("date_id") >= 5)
            .then(pl.col(f"responder_{i}") * 77 + 44)
            .otherwise(pl.col(f"responder_{i}"))
            .alias(f"responder_{i}")
            for i in range(9)
        ]
    )
    first = APISimulator(source, [3, 4, 5]).run(a.predict).predictions
    second = APISimulator(FrameSource(altered), [3, 4, 5]).run(b.predict).predictions
    np.testing.assert_array_equal(first["responder_6"], second["responder_6"])
    assert [x["released_at"] for x in a.update_log] == [[4, 0], [5, 0]]
    assert [x["source_date"] for x in a.update_log] == [3, 4]
    assert all(x["steps"] == 3 for x in a.update_log)
    for p, q in zip(a.models[0].parameters(), b.models[0].parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert a.scaler.state_dict() == prepared.scaler.state_dict()


def test_streaming_cache_matches_full_day_after_symbol_reordering(tmp_path):
    source, _, _ = train(tmp_path)
    predictor = load_predictor(tmp_path / "model")
    predictor.config = replace(predictor.config, enabled=False)
    day = source.day(5)  # Contains a new symbol and a missing intraday observation.
    module = api()
    arrays, keys = module.pack_panel(public_view(day), predictor.features)
    with torch.no_grad():
        expected, _ = predictor.models[0](*[torch.as_tensor(a) for a in arrays[:3]])
    expected_by_key = {tuple(k): float(expected[0, t, s, 6]) for k, t, s in keys}
    outputs = []
    for batch in public_view(day).partition_by("time_id", maintain_order=True):
        batch = batch.reverse()
        pred = predictor.predict(batch, None)
        outputs.extend(pred["responder_6"].to_list())
        want = [
            expected_by_key[tuple(k)]
            for k in batch.select("date_id", "time_id", "symbol_id").iter_rows()
        ]
        np.testing.assert_allclose(pred["responder_6"], want, rtol=1e-5, atol=1e-6)
    assert np.isfinite(outputs).all()


def test_online_seed_ensemble_equals_separate_models(tmp_path):
    import copy

    source, _, config = train(tmp_path)
    first = load_predictor(tmp_path / "model")
    second = load_predictor(tmp_path / "model")
    with torch.no_grad():
        next(second.models[0].parameters()).add_(0.02)
    cls = api().PatrickPredictor
    separate = [
        cls([copy.deepcopy(p.models[0])], copy.deepcopy(p.features), config.online, [seed])
        for seed, p in zip([3, 9], [first, second], strict=True)
    ]
    ensemble = cls([first.models[0], second.models[0]], first.features, config.online, [3, 9])
    actual = (
        APISimulator(source, [3, 4, 5]).run(ensemble.predict).predictions["responder_6"].to_numpy()
    )
    expected = np.mean(
        [
            APISimulator(source, [3, 4, 5]).run(p.predict).predictions["responder_6"].to_numpy()
            for p in separate
        ],
        axis=0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
