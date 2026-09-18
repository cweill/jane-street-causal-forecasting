import copy
import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import torch
from test_patrick_pipeline import panel, train

from src.artifacts import load_predictor
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.models.patrick_yam import PatrickModelConfig, PatrickYam
from src.training.checkpoints import load_replay_checkpoint, save_replay_checkpoint
from src.training.patrick import PatrickPredictor


def stacked_api():
    assert importlib.util.find_spec("src.models.patrick_ensemble"), "stacked ensemble missing"
    return importlib.import_module("src.models.patrick_ensemble")


@pytest.mark.parametrize("post_norm", [False, True])
def test_batched_predictions_and_states_match_distinct_models_and_refresh(post_norm):
    api = stacked_api()
    config = PatrickModelConfig(
        d_model=8,
        nheads=2,
        d_hidden=16,
        layers=2,
        rnn_multiplier=2,
        head_sizes=(12, 8),
        post_norm=post_norm,
    )
    models = []
    for seed in (2, 5, 7):
        torch.manual_seed(seed)
        models.append(PatrickYam(config, (5, 5, 5)).eval())
    engine = api.StackedPatrick(models)
    reference_states = [None] * len(models)
    states = None
    with torch.no_grad():
        for _ in range(8):
            x, cats = torch.randn(4, 77), torch.randint(0, 5, (4, 3))
            prediction, states = engine.forward_step(x, cats, states)
            assert prediction.shape == (3, 4, 9)
            assert not prediction.requires_grad and not states.requires_grad
            for i, model in enumerate(models):
                expected, reference_states[i] = model.forward_step(
                    x[None, None], cats[None, None], reference_states[i]
                )
                torch.testing.assert_close(prediction[i], expected[0, 0], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(
                    states[i], torch.cat(reference_states[i]), rtol=2e-5, atol=2e-6
                )
        # A refresh must pick up newly trained weights rather than serve stale copies.
        models[1].head[-1].bias.add_(0.75)
        engine.refresh(models)
        actual, _ = engine.forward_step(x, cats)
        expected, _ = models[1].forward_step(x[None, None], cats[None, None])
        torch.testing.assert_close(actual[1], expected[0, 0], rtol=2e-5, atol=2e-6)


def ensemble_predictor(tmp_path, stacked):
    base = load_predictor(tmp_path / "model")
    models = [copy.deepcopy(base.models[0]) for _ in range(3)]
    with torch.no_grad():
        models[1].head[-1].bias.add_(0.2)
        models[2].blocks[0].projection.bias.sub_(0.1)
    return PatrickPredictor(
        models, base.features, base.config, seeds=(3, 9, 17), stacked_inference=stacked
    )


def test_stacked_replay_preserves_updates_missing_symbols_and_resume(tmp_path):
    source, _, _ = train(tmp_path)
    assert "stacked_inference" in __import__("inspect").signature(PatrickPredictor).parameters
    slow, fast = ensemble_predictor(tmp_path, False), ensemble_predictor(tmp_path, True)
    a = APISimulator(source, [3, 4]).run(slow.predict)
    b = APISimulator(source, [3, 4]).run(fast.predict)
    np.testing.assert_allclose(
        a.predictions["responder_6"], b.predictions["responder_6"], rtol=2e-5, atol=2e-6
    )
    save_replay_checkpoint(
        fast, tmp_path / "resume", signature="ensemble", row_offset=b.predictions.height
    )
    restored, _ = load_replay_checkpoint(tmp_path / "resume", signature="ensemble", device="cpu")
    assert restored.stacked_inference
    expected = APISimulator(source, [5, 6]).run(slow.predict).predictions
    actual = APISimulator(source, [5, 6]).run(fast.predict).predictions
    resumed = APISimulator(source, [5, 6]).run(restored.predict).predictions
    np.testing.assert_allclose(expected["responder_6"], actual["responder_6"], rtol=2e-5, atol=2e-6)
    assert actual.equals(resumed)
    assert fast.update_log == slow.update_log == restored.update_log
    for a, b in zip(slow.models, fast.models, strict=True):
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    for a, b in zip(slow._optimizers, fast._optimizers, strict=True):
        for key, state in a.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, b.state_dict()["state"][key][name], rtol=0, atol=0
                )


def test_future_responders_cannot_affect_stacked_predictions_or_updates(tmp_path):
    source, _, _ = train(tmp_path)
    assert "stacked_inference" in __import__("inspect").signature(PatrickPredictor).parameters
    clean, changed = ensemble_predictor(tmp_path, True), ensemble_predictor(tmp_path, True)
    poisoned = panel().with_columns(
        [
            pl.when(pl.col("date_id") >= 5)
            .then(999.0)
            .otherwise(pl.col(f"responder_{i}"))
            .alias(f"responder_{i}")
            for i in range(9)
        ]
    )
    a = APISimulator(source, [3, 4, 5]).run(clean.predict).predictions
    b = APISimulator(FrameSource(poisoned), [3, 4, 5]).run(changed.predict).predictions
    assert a.equals(b)
    for x, y in zip(clean.models, changed.models, strict=True):
        for name, value in x.state_dict().items():
            torch.testing.assert_close(value, y.state_dict()[name], rtol=0, atol=0)


def test_stacking_rejects_incompatible_architectures():
    api = stacked_api()
    config = PatrickModelConfig(d_model=8, nheads=2, d_hidden=16, layers=1, head_sizes=(8,))
    a = PatrickYam(config, (5, 5, 5))
    b = PatrickYam(replace(config, nheads=4), (5, 5, 5))
    with pytest.raises(ValueError, match="homogeneous"):
        api.StackedPatrick([a, b])


def test_config_switch_reaches_saved_predictor_and_shared_replay(tmp_path, monkeypatch):
    from test_patrick_pipeline import tiny_config

    from src.config import from_dict
    from src.experiment import run_experiment

    raw = tiny_config().to_dict()
    raw["inference"] = {"stacked_ensemble": True}
    raw["ensemble"].update(seed_ensembling=True, seeds=[2, 5])
    raw["cv"].update(min_date=0, gap_days=1, validation_days=2)
    config = from_dict(raw)
    assert config.inference.stacked_ensemble
    monkeypatch.setattr("src.experiment.safety_gate", lambda: {"passed": True})
    run_experiment(FrameSource(panel()), config, tmp_path / "run")
    restored = load_predictor(tmp_path / "run/fold_0/checkpoint")
    assert restored.stacked_inference and len(restored.models) == 2
    with pytest.raises(ValueError, match="boolean"):
        from_dict({**raw, "inference": {"stacked_ensemble": "yes"}})
