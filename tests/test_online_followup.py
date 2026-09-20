import importlib
import importlib.util
import json

import polars as pl
import pytest
from test_online_sweep import fixture

from src.artifacts import sha256_file
from src.parallel_replay import run_replay_mode


def api():
    assert importlib.util.find_spec("src.online_followup"), "online follow-up runner missing"
    return importlib.import_module("src.online_followup")


def reference(tmp_path):
    source, checkpoint, fold = fixture(tmp_path)
    root = tmp_path / "reference"
    root.mkdir()
    checkpoint.rename(root / "initial_checkpoint")
    for mode in ("offline", "online"):
        run_replay_mode(
            source,
            root / mode,
            checkpoint=root / "initial_checkpoint",
            mode=mode,
            replay_dates=fold.replay_dates,
            scored_dates=fold.validation_dates,
            device="cpu",
            provenance={"source": "fixed"},
        )
    return source, root, fold


def test_followup_reuses_weights_changes_only_lr_and_compares_three_modes(tmp_path):
    module = api()
    source, original, fold = reference(tmp_path)
    root = tmp_path / "followup"
    record = module.prepare_followup(original, root, 1e-4)
    old = json.loads((original / "initial_checkpoint/metadata.json").read_text())
    new = json.loads((root / "initial_checkpoint/metadata.json").read_text())
    old["online"]["learning_rate"] = 1e-4
    assert new == old
    assert sha256_file(root / "initial_checkpoint/weights.pt") == sha256_file(
        original / "initial_checkpoint/weights.pt"
    )
    assert record["replay_dates"] == list(fold.replay_dates)
    assert record["scored_dates"] == list(fold.validation_dates)
    result = run_replay_mode(
        source,
        root / "online",
        checkpoint=root / "initial_checkpoint",
        mode="online",
        replay_dates=fold.replay_dates,
        scored_dates=fold.validation_dates,
        device="cpu",
        provenance=record,
    )
    module.check_first_day(root)
    summary = module.finish_followup(root, window=2)
    assert summary["online_1e-4"]["score"] == result["score"]
    assert summary["delta_from_frozen"] == result["score"] - summary["frozen"]["score"]
    assert (root / "comparison.png").exists()
    score_path = root / "online/result.json"
    score_text = score_path.read_text()
    corrupted = json.loads(score_text)
    corrupted["scored_rows"] += 1
    score_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="coverage"):
        module.finish_followup(root, window=2)
    score_path.write_text(score_text)
    assert module.prepare_followup(original, root, 1e-4) == record
    with pytest.raises(ValueError, match="identity"):
        module.prepare_followup(original, root, 1e-3)
    p = root / "online/date_3.parquet"
    pl.read_parquet(p).with_columns(pl.col("responder_6") + 1).write_parquet(p)
    with pytest.raises(ValueError, match="first-day"):
        module.check_first_day(root)


def test_followup_rejects_mismatched_baseline_and_training_boundary(tmp_path):
    module = api()
    _, original, _ = reference(tmp_path)
    path = original / "offline/identity.json"
    identity = json.loads(path.read_text())
    identity["scored_dates"] = [5, 6]
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="reference"):
        module.prepare_followup(original, tmp_path / "bad", 1e-4)
    identity["scored_dates"] = [4, 5, 6]
    path.write_text(json.dumps(identity))
    meta = original / "initial_checkpoint/metadata.json"
    data = json.loads(meta.read_text())
    data["feature_state"]["scaler"]["training_dates"] += [3]
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reference|training"):
        module.prepare_followup(original, tmp_path / "bad2", 1e-4)
