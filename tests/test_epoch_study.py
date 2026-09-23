import importlib
import importlib.util
from dataclasses import replace

import pytest
import torch
from test_epoch_validation import assert_tree_equal
from test_patrick_pipeline import panel, tiny_config

from src.artifacts import load_predictor
from src.data.loader import FrameSource
from src.training.patrick import prepare_training, train_model
from src.training.validation import prepare_validation


def api():
    assert importlib.util.find_spec("src.epoch_study"), "epoch budget study missing"
    return importlib.import_module("src.epoch_study")


@pytest.mark.parametrize("captures", [(3, 4), (5, 7, 9)])
def test_epoch_snapshots_preserve_training_rng_and_recover_missing_observation(tmp_path, captures):
    mod = api()
    import inspect

    assert "capture_epochs" in inspect.signature(mod.snapshot_epoch).parameters
    budget = max(captures)
    c = tiny_config()
    c = replace(
        c,
        model=replace(c.model, dropout=0.2),
        training=replace(c.training, epochs=budget),
        online=replace(c.online, learning_rate=1e-4),
    )
    source = FrameSource(panel())
    prepared = prepare_training(source, (0, 1, 2), c.features, tmp_path / "train")
    val = prepare_validation(source, (4, 5, 6), prepared, tmp_path / "val")
    reference = tmp_path / "reference.pt"
    model, history = train_model(
        prepared, c.model, training=c.training, seed=0, checkpoint_path=reference, validation=val
    )
    checkpoint = tmp_path / "observed.pt"
    snapshots = tmp_path / "snapshots"

    def observer(record):
        if not record["checkpoint_written"]:
            return
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["epoch"] == captures[0] and saved["position"] == 0 and not snapshots.exists():
            raise InterruptedError("after checkpoint, before epoch artifact")
        mod.snapshot_epoch(saved, prepared, c, 0, snapshots, capture_epochs=captures)

    with pytest.raises(InterruptedError):
        train_model(
            prepared,
            c.model,
            training=c.training,
            seed=0,
            checkpoint_path=checkpoint,
            validation=val,
            progress=observer,
        )
    snapshots.mkdir()
    _actual, actual_history = train_model(
        prepared,
        c.model,
        training=c.training,
        seed=0,
        checkpoint_path=checkpoint,
        validation=val,
        progress=observer,
    )
    assert actual_history == history
    a = torch.load(reference, weights_only=True)
    b = torch.load(checkpoint, weights_only=True)
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "numpy_rng"):
        assert_tree_equal(a[key], b[key])
    for epoch in captures:
        assert load_predictor(snapshots / f"epoch_{epoch}/checkpoint").seeds == (0,)
    saved = load_predictor(snapshots / f"epoch_{budget}/checkpoint")
    assert_tree_equal(saved.models[0].state_dict(), model.state_dict())
    assert saved.config.learning_rate == 1e-4
    mod.check_history(history, history)
    mod.check_history(history[:3], history)
    _, longer_history = train_model(
        prepared,
        c.model,
        training=replace(c.training, epochs=budget + 1),
        seed=0,
        validation=val,
    )
    assert history == longer_history[:budget]
    bad = [dict(h) for h in history]
    bad[0]["responder_6_r2"] += 0.01
    with pytest.raises(ValueError, match="history"):
        mod.check_history(history, bad)
    with pytest.raises(ValueError, match="identity"):
        mod.snapshot_epoch(
            b,
            prepared,
            replace(c, online=replace(c.online, learning_rate=5e-4)),
            0,
            snapshots,
            capture_epochs=captures,
        )


def test_epoch_comparison_pairs_within_epoch_and_rejects_mismatched_weights(tmp_path):
    mod = api()
    import json

    from src.artifacts import save_predictor, write_json
    from src.cv import TemporalFold
    from src.parallel_replay import run_replay_mode

    c = tiny_config()
    source = FrameSource(panel())
    prepared = prepare_training(source, (0, 1, 2), c.features, tmp_path / "train")
    fold = TemporalFold(0, (0, 1, 2), (3,), (4, 5, 6))
    roots = {}
    for epoch in (5, 7, 9):
        root = tmp_path / f"e{epoch}"
        root.mkdir()
        model, _ = train_model(
            prepared, c.model, training=replace(c.training, epochs=epoch), seed=0
        )
        save_predictor(
            root / "initial_checkpoint",
            [model],
            prepared.features,
            prepared.scaler,
            replace(c.online, learning_rate=1e-4),
            [0],
        )
        roots[epoch] = {}
        for mode in ("offline", "online"):
            roots[epoch][mode] = root / mode
            run_replay_mode(
                source,
                root / mode,
                checkpoint=root / "initial_checkpoint",
                mode=mode,
                replay_dates=fold.replay_dates,
                scored_dates=fold.validation_dates,
                device="cpu",
                provenance={"dataset_sha256": "same"},
            )
    report = mod.compare_epochs(roots, fold, tmp_path / "report", window=2)
    assert [r["epoch"] for r in report["epochs"]] == [5, 7, 9]
    assert all(r["online_r2"] - r["frozen_r2"] == r["delta_r2"] for r in report["epochs"])
    assert (tmp_path / "report/rolling_comparison.png").exists()
    p = roots[5]["online"] / "result.json"
    d = json.loads(p.read_text())
    d["initial_weights_sha256"] = "different"
    write_json(p, d)
    with pytest.raises(ValueError, match="paired"):
        mod.compare_epochs(roots, fold, tmp_path / "bad", window=2)


def test_confirmation_protocol_rejects_parameter_drift_and_missing_dates():
    from src.config import load_config

    mod = api()
    assert hasattr(mod, "confirmation_fold"), "fixed confirmation protocol missing"
    reference = load_config("configs/patrick_long_epochs.yaml")
    config = replace(
        reference,
        name="patrick_epoch_confirmation",
        cv=replace(reference.cv, train_end=859, warmup_end=979, validation_end=1179),
    )
    dates = tuple(range(1699))
    fold = mod.confirmation_fold(config, reference, dates)
    assert fold.train_dates == tuple(range(860))
    assert fold.warmup_dates == tuple(range(860, 980))
    assert fold.validation_dates == tuple(range(980, 1180))
    for changed in (
        replace(config, online=replace(config.online, learning_rate=5e-4)),
        replace(config, training=replace(config.training, learning_rate=1e-4)),
        replace(config, ensemble=replace(config.ensemble, seeds=(3, 4, 5))),
        replace(config, cv=reference.cv),
        replace(config, features=replace(config.features, time_epsilon=0.01)),
    ):
        with pytest.raises(ValueError, match="confirmation"):
            mod.confirmation_fold(changed, reference, dates)
    with pytest.raises(ValueError, match="missing"):
        mod.confirmation_fold(config, reference, [d for d in dates if d != 979])
