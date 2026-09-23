import importlib
import importlib.util
import json
from dataclasses import replace

import pytest
import torch
from test_patrick_pipeline import panel, tiny_config

from src.artifacts import save_predictor, write_json
from src.config import load_config
from src.cv import TemporalFold
from src.data.loader import FrameSource
from src.parallel_replay import run_replay_mode
from src.training.patrick import prepare_training, train_model


def api():
    assert importlib.util.find_spec("src.full_ensemble"), "staged full ensemble support missing"
    return importlib.import_module("src.full_ensemble")


def test_fixed_full_protocol_and_remaining_seeds_require_matching_pilot():
    m = api()
    reference = load_config("configs/patrick_ensemble.yaml")
    config = replace(
        reference,
        name="patrick_nine_epoch_ensemble",
        training=replace(reference.training, epochs=9),
        online=replace(reference.online, learning_rate=1e-4),
    )
    f = m.full_fold(config, reference, range(1699))
    assert f.train_dates == tuple(range(1380))
    assert f.warmup_dates == tuple(range(1380, 1500))
    assert f.validation_dates == tuple(range(1500, 1699))
    with pytest.raises(ValueError, match="protocol"):
        m.full_fold(
            replace(config, training=replace(config.training, learning_rate=1e-4)),
            reference,
            range(1699),
        )
    m.require_seed_stage(0, None, "identity")
    with pytest.raises(ValueError, match="pilot"):
        m.require_seed_stage(3, None, "identity")
    with pytest.raises(ValueError, match="pilot"):
        m.require_seed_stage(3, {"passed": True, "code_sha256": "different"}, "identity")
    m.require_seed_stage(16, {"passed": True, "code_sha256": "identity"}, "identity")
    with pytest.raises(ValueError, match="seed"):
        m.require_seed_stage(17, {"passed": True, "code_sha256": "identity"}, "identity")


def test_epoch_five_control_rejects_changed_weights():
    m = api()
    saved = {"epoch": 5, "position": 0, "model": {"w": torch.tensor([1.0])}}
    assert m.check_epoch_five(saved, {"w": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="epoch-five"):
        m.check_epoch_five(saved, {"w": torch.tensor([2.0])})
    assert not m.check_epoch_five({**saved, "epoch": 4}, {"w": torch.tensor([2.0])})


def paired(tmp_path, epoch):
    c = tiny_config()
    c = replace(c, online=replace(c.online, learning_rate=1e-4, steps=3))
    source = FrameSource(panel())
    p = prepare_training(source, (0, 1, 2), c.features, tmp_path / "train")
    model, _ = train_model(p, c.model, training=replace(c.training, epochs=epoch), seed=0)
    save_predictor(tmp_path / "initial_checkpoint", [model], p.features, p.scaler, c.online, [0])
    fold = TemporalFold(0, (0, 1, 2), (3,), (4, 5, 6))
    for mode in ("offline", "online"):
        run_replay_mode(
            source,
            tmp_path / mode,
            checkpoint=tmp_path / "initial_checkpoint",
            mode=mode,
            replay_dates=fold.replay_dates,
            scored_dates=fold.validation_dates,
            device="cpu",
            provenance={"dataset_sha256": "fixture"},
        )
    return fold


def test_paired_gate_and_baseline_comparison_reject_leakage_or_coverage_changes(tmp_path):
    m = api()
    old, new = tmp_path / "old", tmp_path / "new"
    fold = paired(old, 1)
    paired(new, 2)
    report = m.verify_pair(new, fold.replay_dates, fold.validation_dates, (0,))
    assert report["passed"] and report["updates"] == 3
    summary = m.compare_baseline(
        new,
        {k: old / k for k in ("offline", "online")},
        fold,
        tmp_path / "report",
        seeds=(0,),
        window=2,
    )
    assert summary["online_delta"] == summary["nine_online"] - summary["five_online"]
    assert (tmp_path / "report/comparison.png").exists()
    path = new / "online/result.json"
    original = json.loads(path.read_text())
    bad = json.loads(path.read_text())
    bad["updates"][0]["source_date"] += 1
    write_json(path, bad)
    with pytest.raises(ValueError, match="update"):
        m.verify_pair(new, fold.replay_dates, fold.validation_dates, (0,))
    write_json(path, original)
    path = old / "online/result.json"
    bad = json.loads(path.read_text())
    bad["scored_rows"] += 1
    write_json(path, bad)
    with pytest.raises(ValueError, match="coverage"):
        m.compare_baseline(
            new,
            {k: old / k for k in ("offline", "online")},
            fold,
            tmp_path / "bad",
            seeds=(0,),
            window=2,
        )


def test_live_replay_metrics_exclude_warmup_and_pool_sufficient_statistics():
    m = api()
    assert hasattr(m, "replay_metrics"), "incremental replay metrics missing"
    rows = [
        {
            "date_id": 10,
            "seconds": 2,
            "diagnostic": {"sse": 9.0, "denominator": 10.0},
            "primary": {"sse": 0.0, "denominator": 0.0, "rows": 0},
        },
        {
            "date_id": 11,
            "seconds": 3,
            "diagnostic": {"sse": 1.0, "denominator": 2.0},
            "primary": {"sse": 1.0, "denominator": 2.0, "rows": 1},
        },
        {
            "date_id": 12,
            "seconds": 4,
            "diagnostic": {"sse": 2.0, "denominator": 8.0},
            "primary": {"sse": 2.0, "denominator": 8.0, "rows": 2},
        },
    ]
    assert "online/scored_cumulative_r2" not in m.replay_metrics(rows[:1], "online", window=2)
    result = m.replay_metrics(rows, "online", window=2)
    assert result["online/scored_cumulative_r2"] == pytest.approx(0.7)
    assert result["online/rolling_r2"] == pytest.approx(0.7)
    assert result["online/date_id"] == 12
    assert result["online/days_completed"] == 3
