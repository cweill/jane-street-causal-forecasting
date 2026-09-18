import importlib
import importlib.util
from dataclasses import replace

import polars as pl
import pytest
from test_patrick_pipeline import panel, tiny_config
from test_resume import DeliberateInterruption

from src.data.loader import FrameSource
from src.reproduction import run_online_comparison


def test_independent_replays_match_sequential_reference_and_resume(tmp_path, monkeypatch):
    assert importlib.util.find_spec("src.parallel_replay"), "independent replay runner missing"
    api = importlib.import_module("src.parallel_replay")
    monkeypatch.setattr(
        "src.reproduction.safety_gate", lambda: {"passed": True, "code_and_tests_sha256": "fixture"}
    )
    config = tiny_config()
    config = replace(
        config,
        cv=replace(
            config.cv,
            min_date=0,
            n_splits=1,
            train_end=2,
            warmup_end=3,
            validation_end=6,
            gap_days=1,
            validation_days=3,
        ),
    )
    source = FrameSource(panel())
    expected = run_online_comparison(
        source,
        config,
        tmp_path / "reference",
        cache_root=tmp_path / "cache",
        dataset_sha256="a" * 64,
        window=2,
    )
    common = {
        "checkpoint": tmp_path / "reference/initial_checkpoint",
        "replay_dates": (3, 4, 5, 6),
        "scored_dates": (4, 5, 6),
        "device": "cpu",
        "provenance": {"source": "fixture"},
        "fast": True,
    }

    def interrupt(record):
        if record.get("completed_date") == 4:
            raise DeliberateInterruption()

    # Independent jobs can finish in either order without sharing state or writing each other's paths.
    with pytest.raises(DeliberateInterruption):
        api.run_replay_mode(
            source, tmp_path / "separate/online", mode="online", progress=interrupt, **common
        )
    actual = {}
    for mode in ("online", "offline"):
        actual[mode] = api.run_replay_mode(
            source, tmp_path / "separate" / mode, mode=mode, **common
        )
        assert actual[mode]["score"] == expected[mode]["score"]
        assert actual[mode]["final_weights_sha256"] == expected[mode]["final_weights_sha256"]
        assert actual[mode]["updates"] == expected[mode]["updates"]
        for date in (3, 4, 5, 6):
            assert pl.read_parquet(tmp_path / "separate" / mode / f"date_{date}.parquet").equals(
                pl.read_parquet(tmp_path / "reference" / mode / f"date_{date}.parquet")
            )
    # Reuse only committed frozen days, then continue with the same row IDs/state.
    (tmp_path / "reference/offline/checkpoints/date_6/replay.json").rename(
        tmp_path / "reference/offline/checkpoints/date_6/pending.json"
    )
    reused = api.run_replay_mode(
        source,
        tmp_path / "reuse/offline",
        mode="offline",
        source_prefix=tmp_path / "reference/offline",
        **common,
    )
    assert reused["reused_days"] == 3
    assert reused["score"] == expected["offline"]["score"]
    for date in (3, 4, 5, 6):
        assert pl.read_parquet(tmp_path / "reuse/offline" / f"date_{date}.parquet").equals(
            pl.read_parquet(tmp_path / "reference/offline" / f"date_{date}.parquet")
        )
    with pytest.raises(ValueError, match="identity"):
        api.run_replay_mode(
            source,
            tmp_path / "separate/online",
            mode="online",
            **{**common, "scored_dates": (5, 6)},
        )
    summary = api.finish_comparison(tmp_path / "separate", window=2, scored_start=4)
    assert summary["status"] == "complete"
    assert (tmp_path / "separate/online_learning.png").exists()
    # Never combine different initial models into an OL comparison.
    import json

    path = tmp_path / "separate/online/result.json"
    damaged = json.loads(path.read_text())
    damaged["initial_weights_sha256"] = "different"
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="initial"):
        api.finish_comparison(tmp_path / "separate", window=2, scored_start=4)
