import copy
import importlib
import importlib.util
import inspect
from dataclasses import replace

import polars as pl
import pytest
import torch
from test_patrick_pipeline import panel, tiny_config

from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource, RestrictedDateSource
from src.monitoring import training_events
from src.training.patrick import (
    PatrickOnlineConfig,
    PatrickPredictor,
    prepare_training,
    train_model,
)


def validation_api():
    assert importlib.util.find_spec("src.training.validation"), "epoch validation missing"
    return importlib.import_module("src.training.validation")


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_epoch_validation_matches_streaming_metric_and_scored_row_mask(tmp_path):
    # Catches daily-R² averaging, unscored-row inclusion, wrong weights, or label inputs.
    api = validation_api()
    frame = panel().with_columns((pl.col("time_id") != 1).alias("is_scored"))
    source, config = FrameSource(frame), tiny_config()
    prepared = prepare_training(source, (0, 1, 2), config.features, tmp_path / "train")
    state = copy.deepcopy(prepared.features.state_dict())
    validation = api.prepare_validation(
        RestrictedDateSource(source, (3, 4)), (3, 4), prepared, tmp_path / "val"
    )
    model, _ = train_model(prepared, config.model, training=config.training)
    want = (
        APISimulator(source, (3, 4))
        .run(
            PatrickPredictor(
                [copy.deepcopy(model)],
                copy.deepcopy(prepared.features),
                PatrickOnlineConfig(enabled=False),
                [0],
            ).predict
        )
        .metric
    )
    model.train()
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    actual = api.evaluate_validation(model, validation)
    assert actual["r2"] == pytest.approx(want.score, abs=1e-7)
    assert actual["sse"] == pytest.approx(want.sse, rel=1e-6)
    assert actual["denominator"] == pytest.approx(want.denominator, rel=1e-12)
    assert actual["rows"] == want.rows
    assert prepared.features.state_dict() == state
    assert model.training
    assert_tree_equal(model.state_dict(), original)
    assert torch.equal(rng, torch.get_rng_state())


def test_validation_and_poisoned_future_responders_cannot_change_training(tmp_path):
    # Catches validation gradients, RNG consumption, train/val mixing and target leakage.
    assert "validation" in inspect.signature(train_model).parameters, "trainer hook missing"
    api, config = validation_api(), tiny_config()
    config = replace(
        config,
        model=replace(config.model, dropout=0.2),
        training=replace(config.training, epochs=3),
    )
    source = FrameSource(panel())
    prepared = prepare_training(source, (0, 1, 2), config.features, tmp_path / "train")
    changed = FrameSource(
        panel().with_columns(
            [
                pl.when(pl.col("date_id") >= 3)
                .then(pl.col(f"responder_{i}") * -100 + 70)
                .otherwise(pl.col(f"responder_{i}"))
                .alias(f"responder_{i}")
                for i in range(9)
            ]
        )
    )
    validations = [None] + [
        api.prepare_validation(s, (3, 4), prepared, tmp_path / f"v{i}")
        for i, s in enumerate((source, changed))
    ]
    states, histories = [], []
    for i, validation in enumerate(validations):
        checkpoint = tmp_path / f"training{i}.pt"
        _, history = train_model(
            prepared,
            config.model,
            training=config.training,
            seed=7,
            checkpoint_path=checkpoint,
            validation=validation,
        )
        states.append(torch.load(checkpoint, weights_only=True))
        histories.append(history)
    for key in ("model", "optimizer", "torch_rng", "numpy_rng", "cuda_rng"):
        assert_tree_equal(states[0][key], states[1][key])
        assert_tree_equal(states[0][key], states[2][key])
    assert len(histories[1]) == 3
    assert all("validation" in row for row in histories[1])
    assert histories[1][-1]["validation"]["r2"] != histories[2][-1]["validation"]["r2"]
    for row, plain in zip(histories[1], histories[0], strict=True):
        assert {k: v for k, v in row.items() if k != "validation"} == plain
    events = training_events(states[1], days_per_epoch=3, epochs=3)
    assert [e["metrics"]["val/epoch"] for e in events] == [1, 2, 3]
    assert [e["metrics"].get("train/batch") for e in events] == [3, 6, 9]
    assert events[-1]["metrics"]["val/responder_6_r2"] == histories[1][-1]["validation"]["r2"]


def test_validation_resume_keeps_epoch_history_and_rejects_changed_holdout(tmp_path):
    api, config = validation_api(), tiny_config()
    config = replace(config, training=replace(config.training, epochs=3))
    source = FrameSource(panel())
    prepared = prepare_training(source, (0, 1), config.features, tmp_path / "train")
    validation = api.prepare_validation(source, (3, 4), prepared, tmp_path / "val")
    expected, history = train_model(
        prepared, config.model, training=config.training, validation=validation
    )

    def interrupt(record):
        if record["epochs_completed"] == 1:
            raise InterruptedError()

    checkpoint = tmp_path / "training.pt"
    with pytest.raises(InterruptedError):
        train_model(
            prepared,
            config.model,
            training=config.training,
            validation=validation,
            checkpoint_path=checkpoint,
            progress=interrupt,
        )
    actual, resumed = train_model(
        prepared,
        config.model,
        training=config.training,
        validation=validation,
        checkpoint_path=checkpoint,
    )
    assert history == resumed
    assert_tree_equal(expected.state_dict(), actual.state_dict())
    different = api.prepare_validation(source, (4, 5), prepared, tmp_path / "other")
    with pytest.raises(ValueError, match="identity"):
        train_model(
            prepared,
            config.model,
            training=config.training,
            validation=different,
            checkpoint_path=checkpoint,
        )
    with pytest.raises(ValueError, match="identity"):
        train_model(prepared, config.model, training=config.training, checkpoint_path=checkpoint)
    with (tmp_path / "val/3.npz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        train_model(prepared, config.model, training=config.training, validation=validation)


def test_validation_rejects_overlap_or_foreign_scaler_and_handles_zero_energy(tmp_path):
    api, config = validation_api(), tiny_config()
    source = FrameSource(panel())
    prepared = prepare_training(source, (0, 1), config.features, tmp_path / "train")
    with pytest.raises(ValueError, match="after training"):
        api.prepare_validation(source, (1, 2), prepared, tmp_path / "overlap")
    zero = FrameSource(panel().with_columns(pl.lit(0.0).alias("responder_6")))
    validation = api.prepare_validation(zero, (3, 4), prepared, tmp_path / "val")
    other = prepare_training(source, (0, 1, 2), config.features, tmp_path / "other")
    with pytest.raises(ValueError, match="preprocessing|training"):
        train_model(other, config.model, training=config.training, validation=validation)
    model, history = train_model(
        prepared, config.model, training=config.training, validation=validation
    )
    assert history[0]["validation"]["r2"] is None
    assert history[0]["validation"]["denominator"] == 0
    assert api.evaluate_validation(model, validation)["r2"] is None


def test_development_runner_excludes_gap_and_later_dates_logs_each_epoch_and_resumes(tmp_path):
    # Catches evaluating the final replay period, fitting validation, or losing epoch artifacts/logs.
    assert importlib.util.find_spec("src.development"), "development entry point missing"
    module = importlib.import_module("src.development")
    config = tiny_config()
    config = replace(
        config,
        training=replace(config.training, epochs=2),
        online=replace(config.online, enabled=False),
        cv=replace(
            config.cv, train_end=1, warmup_end=2, validation_end=4, gap_days=1, validation_days=2
        ),
    )

    class BoundedSource:
        def __init__(self):
            self.source, self.reads = FrameSource(panel()), []

        def dates(self):
            return self.source.dates()

        def day(self, date):
            assert date in (0, 1, 3, 4), "gap or later feature/target rows were read"
            self.reads.append(date)
            return self.source.day(date)

    # A collector substitutes only the external W&B network boundary.
    class Tracker:
        def __init__(self):
            self.events, self.summary = [], {}

        def log(self, metrics, step):
            self.events.append((step, metrics))

    source, tracker = BoundedSource(), Tracker()
    output = tmp_path / "run"
    result = module._run_development_verified(
        source, config, output, {"passed": True}, tracker=tracker
    )
    val_events = [m for _, m in tracker.events if "val/epoch" in m]
    assert [m["val/epoch"] for m in val_events] == [1, 2]
    assert result["validation_r2"] == val_events[-1]["val/responder_6_r2"]
    assert (output / "epochs/epoch_1/metadata.json").is_file()
    assert (output / "epochs/epoch_2/metadata.json").is_file()
    before = len(source.reads)
    resumed = module._run_development_verified(
        source, config, output, {"passed": True}, tracker=tracker, resume=True
    )
    assert resumed == result and len(source.reads) == before
    assert len([m for _, m in tracker.events if "val/epoch" in m]) == 2
    wrong = replace(config, cv=replace(config.cv, validation_end=5, validation_days=3))
    with pytest.raises(ValueError, match="identity"):
        module._run_development_verified(source, wrong, output, {"passed": True}, resume=True)


def test_development_rejects_current_replay_period_before_reading_rows(tmp_path):
    assert importlib.util.find_spec("src.development"), "development entry point missing"
    module = importlib.import_module("src.development")
    from src.config import load_config

    config = load_config("configs/patrick_epoch_validation.yaml")
    invalid = replace(config, cv=replace(config.cv, validation_end=1380, validation_days=201))

    class NoRows:
        def dates(self):
            return tuple(range(1699))

        def day(self, date):
            raise AssertionError("must reject split before reading rows")

    with pytest.raises(ValueError, match="1380"):
        module._run_development_verified(NoRows(), invalid, tmp_path / "run", {"passed": True})


def test_development_recovers_epoch_snapshot_if_interrupted_before_observation(
    tmp_path, monkeypatch
):
    module = importlib.import_module("src.development")
    config = tiny_config()
    config = replace(
        config,
        model=replace(config.model, dropout=0.2),
        training=replace(config.training, epochs=2),
        online=replace(config.online, enabled=False),
        cv=replace(
            config.cv, train_end=1, warmup_end=2, validation_end=4, gap_days=1, validation_days=2
        ),
    )
    source, output = FrameSource(panel()), tmp_path / "run"
    original = module.train_model

    def interrupted(*args, **kwargs):
        observer = kwargs["progress"]

        def stop(record):
            if record["epochs_completed"] == 1:
                raise InterruptedError("after checkpoint, before observer")
            observer(record)

        return original(*args, **{**kwargs, "progress": stop})

    monkeypatch.setattr(module, "train_model", interrupted)
    with pytest.raises(InterruptedError):
        module._run_development_verified(source, config, output, {"passed": True})
    assert not (output / "epochs/epoch_1").exists()
    monkeypatch.setattr(module, "train_model", original)
    module._run_development_verified(source, config, output, {"passed": True}, resume=True)
    assert (output / "epochs/epoch_1/metadata.json").exists()
    prepared = prepare_training(source, (0, 1), config.features, tmp_path / "reference")
    original(prepared, config.model, training=config.training, checkpoint_path=tmp_path / "ref.pt")
    expected = torch.load(tmp_path / "ref.pt", weights_only=True)
    actual = torch.load(output / "training.pt", weights_only=True)
    for key in ("model", "optimizer", "torch_rng", "numpy_rng"):
        assert_tree_equal(expected[key], actual[key])
