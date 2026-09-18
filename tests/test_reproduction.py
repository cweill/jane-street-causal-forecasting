import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from test_patrick_pipeline import panel, tiny_config
from test_resume import DeliberateInterruption

from src.data.loader import FrameSource


def test_paired_reproduction_survives_interruption_and_future_labels_stay_private(
    tmp_path, monkeypatch
):
    assert importlib.util.find_spec("src.reproduction"), "paired resumable plot runner missing"
    module = importlib.import_module("src.reproduction")
    monkeypatch.setattr(
        module, "safety_gate", lambda: {"passed": True, "code_and_tests_sha256": "fixture"}
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
    common = {"cache_root": tmp_path / "cache", "dataset_sha256": "a" * 64, "window": 2}
    first = module.run_online_comparison(source, config, tmp_path / "whole", **common)

    def interrupt(progress):
        if progress.get("phase") == "online" and progress.get("completed_date") == 4:
            raise DeliberateInterruption()

    with pytest.raises(DeliberateInterruption):
        module.run_online_comparison(
            source, config, tmp_path / "resumed", progress=interrupt, **common
        )
    second = module.run_online_comparison(
        source, config, tmp_path / "resumed", resume=True, **common
    )
    for mode in ["offline", "online"]:
        assert first[mode]["score"] == second[mode]["score"]
        assert first[mode]["final_weights_sha256"] == second[mode]["final_weights_sha256"]
        for date in [3, 4, 5, 6]:
            a = pl.read_parquet(tmp_path / "whole" / mode / f"date_{date}.parquet")
            b = pl.read_parquet(tmp_path / "resumed" / mode / f"date_{date}.parquet")
            assert a.equals(b)
    assert Path(second["plot"]["png"]).exists()
    assert first["offline"]["initial_weights_sha256"] == first["online"]["initial_weights_sha256"]
    altered = panel().with_columns(
        [
            pl.when(pl.col("date_id") == 6)
            .then(999.0)
            .otherwise(pl.col(f"responder_{i}"))
            .alias(f"responder_{i}")
            for i in range(9)
        ]
    )
    poison = module.run_online_comparison(
        FrameSource(altered), config, tmp_path / "poison", **{**common, "dataset_sha256": "b" * 64}
    )
    assert poison["online"]["final_weights_sha256"] == first["online"]["final_weights_sha256"]
    for date in [3, 4, 5, 6]:
        assert pl.read_parquet(tmp_path / "whole/online" / f"date_{date}.parquet").equals(
            pl.read_parquet(tmp_path / "poison/online" / f"date_{date}.parquet")
        )
    with pytest.raises(ValueError, match="identity"):
        module.run_online_comparison(
            source,
            replace(config, online=replace(config.online, learning_rate=0.1)),
            tmp_path / "resumed",
            resume=True,
            **common,
        )
