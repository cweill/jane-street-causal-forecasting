import importlib
import importlib.util
import inspect
from dataclasses import replace

import polars as pl
import pytest
import torch
from test_patrick_pipeline import panel, tiny_config, train

from src.artifacts import load_predictor
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.training.patrick import prepare_training, train_model


class DeliberateInterruption(Exception):
    pass


def test_interrupted_training_restores_optimizer_shuffle_position_and_rng(tmp_path):
    assert "checkpoint_path" in inspect.signature(train_model).parameters, "training resume missing"
    config = tiny_config()
    config = replace(
        config,
        model=replace(config.model, dropout=0.15),
        training=replace(config.training, epochs=3),
    )
    prepared = prepare_training(FrameSource(panel()), (0, 1, 2), config.features, tmp_path / "data")
    expected, expected_history = train_model(
        prepared, config.model, training=config.training, seed=9
    )

    def interrupt(progress):
        if progress["completed_batches"] == 4:
            raise DeliberateInterruption()

    checkpoint = tmp_path / "training.pt"
    with pytest.raises(DeliberateInterruption):
        train_model(
            prepared,
            config.model,
            training=config.training,
            seed=9,
            checkpoint_path=checkpoint,
            checkpoint_every=2,
            progress=interrupt,
        )
    actual, history = train_model(
        prepared,
        config.model,
        training=config.training,
        seed=9,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
    )
    assert all("mean_unbalanced_loss" in row and "responder_6_r2" in row for row in history)
    assert history == expected_history
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        train_model(
            prepared,
            config.model,
            training=replace(config.training, learning_rate=1e-3),
            seed=9,
            checkpoint_path=checkpoint,
        )


def checkpoint_api():
    assert importlib.util.find_spec("src.training.checkpoints"), "replay checkpoint support missing"
    return importlib.import_module("src.training.checkpoints")


def test_daily_resume_preserves_online_optimizer_and_public_input_cache(tmp_path):
    module = checkpoint_api()
    source, _, _ = train(tmp_path)
    uninterrupted = load_predictor(tmp_path / "model")
    first = APISimulator(source, [3, 4]).run(uninterrupted.predict).predictions
    path = tmp_path / "day_4"
    module.save_replay_checkpoint(uninterrupted, path, signature="fixture", row_offset=first.height)
    restored, state = module.load_replay_checkpoint(path, signature="fixture", device="cpu")
    assert state["completed_date"] == 4 and state["row_offset"] == first.height
    cached = pl.read_parquet(path / "public_cache.parquet")
    assert not any(c.startswith("responder_") for c in cached.columns)
    expected = APISimulator(source, [5, 6]).run(uninterrupted.predict).predictions
    actual = APISimulator(source, [5, 6]).run(restored.predict).predictions
    assert actual.equals(expected)
    assert restored.update_log == uninterrupted.update_log
    for name, value in uninterrupted.models[0].state_dict().items():
        torch.testing.assert_close(restored.models[0].state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        module.load_replay_checkpoint(path, signature="different", device="cpu")
    illegal, _ = module.load_replay_checkpoint(path, signature="fixture", device="cpu")
    with pytest.raises(ValueError, match="boundary"):
        APISimulator(source, [4]).run(illegal.predict)


def test_day_at_a_time_replay_keeps_global_api_row_ids(panel):
    assert "row_offset" in inspect.signature(APISimulator).parameters, (
        "resumable row offset missing"
    )
    source = FrameSource(panel)

    def zero(test, lags):
        return test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))

    expected = APISimulator(source, [2, 3, 4]).run(zero).predictions
    offset, parts = 0, []
    for date in [2, 3, 4]:
        part = APISimulator(source, [date], row_offset=offset).run(zero).predictions
        offset += part.height
        parts.append(part)
    assert pl.concat(parts).equals(expected)


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_nonfinite_training_keeps_last_healthy_checkpoint(tmp_path, monkeypatch, bad_gradient):
    import src.training.patrick as module

    config = tiny_config()
    prepared = prepare_training(FrameSource(panel()), (0, 1, 2), config.features, tmp_path / "data")
    checkpoint = tmp_path / "training.pt"

    def interrupt(record):
        if record["completed_batches"] == 1:
            raise DeliberateInterruption()

    with pytest.raises(DeliberateInterruption):
        train_model(
            prepared,
            config.model,
            training=config.training,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            progress=interrupt,
        )
    healthy = checkpoint.read_bytes()

    def corrupt_loss(model, prediction, y, w):
        if bad_gradient:
            prediction.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            return prediction.sum()
        return prediction.sum() * float("nan")

    monkeypatch.setattr(module, "loss_for_model", corrupt_loss)
    with pytest.raises((FloatingPointError, RuntimeError), match="non.finite"):
        train_model(
            prepared,
            config.model,
            training=config.training,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
        )
    assert checkpoint.read_bytes() == healthy


def test_nonfinite_online_update_stops_before_mutating_weights(tmp_path, monkeypatch):
    import src.training.patrick as module

    source, _, _ = train(tmp_path)
    predictor = load_predictor(tmp_path / "model")
    APISimulator(source, [3]).run(predictor.predict)
    weights = {name: value.clone() for name, value in predictor.models[0].state_dict().items()}
    monkeypatch.setattr(
        module, "loss_for_model", lambda model, prediction, y, w: prediction.sum() * float("nan")
    )
    with pytest.raises(FloatingPointError, match="non.finite"):
        APISimulator(source, [4]).run(predictor.predict)
    for name, value in weights.items():
        torch.testing.assert_close(predictor.models[0].state_dict()[name], value, rtol=0, atol=0)
    assert predictor.update_log == []
