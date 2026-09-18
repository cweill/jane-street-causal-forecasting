import json
from dataclasses import replace

from test_experiments import differing_paths
from test_patrick_pipeline import panel, tiny_config

from experiments.ablations import ablation_configs
from src.data.loader import FrameSource
from src.experiment import run_experiment


def test_patrick_ablations_change_one_factor_each():
    config = tiny_config()
    variants = list(ablation_configs(config))
    assert len(variants) == 8
    for _, variant in variants[1:]:
        paths = differing_paths(config.to_dict(), variant.to_dict())
        assert len([p for p in paths if p != "name"]) == 1


def test_patrick_runs_through_shared_cv_and_reloads_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.experiment.safety_gate", lambda: {"passed": True, "test_fixture": True}
    )
    config = tiny_config()
    config = replace(config, cv=replace(config.cv, min_date=0, gap_days=1, validation_days=2))
    result = run_experiment(FrameSource(panel()), config, tmp_path / "run")
    assert result["folds"][0]["online_updates"] == 2
    meta = json.loads((tmp_path / "run/fold_0/checkpoint/metadata.json").read_text())
    assert meta["feature_state"]["scaler"]["training_dates"] == [0, 1, 2, 3]


def test_patrick_cli_smoke_runs_complete_synthetic_experiment(tmp_path, monkeypatch):
    from src.cli import main

    monkeypatch.setattr(
        "src.experiment.safety_gate", lambda: {"passed": True, "test_fixture": True}
    )
    output = tmp_path / "smoke"
    monkeypatch.setattr(
        "sys.argv",
        ["js-repro", "smoke", "--config", "configs/patrick.yaml", "--output", str(output)],
    )
    main()
    result = json.loads((output / "result.json").read_text())
    assert len(result["folds"]) == 2
    assert all(fold["online_updates"] > 0 for fold in result["folds"])


def test_fixed_development_runner_never_reads_later_days(tmp_path, monkeypatch):
    from pathlib import Path

    from src.config import load_config

    path = Path("configs/patrick_development.yaml")
    assert path.exists(), "fixed Patrick protocol has not been added"
    pinned = load_config(path)
    config = tiny_config()
    config = replace(
        config,
        cv=replace(
            pinned.cv,
            train_end=2,
            warmup_end=3,
            validation_end=5,
            gap_days=1,
            validation_days=2,
        ),
    )
    source = FrameSource(panel())

    class DevelopmentOnly:
        def dates(self):
            return source.dates()  # Includes held-out date 6; metadata is permitted.

        def day(self, date):
            assert date <= 5, "future data read during development run"
            return source.day(date)

    monkeypatch.setattr("src.experiment.safety_gate", lambda: {"passed": True})
    result = run_experiment(DevelopmentOnly(), config, tmp_path / "fixed")
    assert result["folds"][0]["online_updates"] == 2
    (split,) = json.loads((tmp_path / "fixed/splits.json").read_text())
    assert split["train_dates"] == [0, 1, 2]
    assert split["warmup_dates"] == [3]
    assert split["validation_dates"] == [4, 5]
    meta = json.loads((tmp_path / "fixed/fold_0/checkpoint/metadata.json").read_text())
    assert meta["feature_state"]["scaler"]["training_dates"] == [0, 1, 2]


def test_shared_runner_reuses_preparation_cache_across_runs(tmp_path, monkeypatch):
    import inspect

    from src.data.loader import ParquetSource

    assert "cache_root" in inspect.signature(run_experiment).parameters, (
        "runner cache option missing"
    )
    path = tmp_path / "data.parquet"
    panel().write_parquet(path)
    source = ParquetSource(path)
    config = tiny_config()
    config = replace(config, cv=replace(config.cv, min_date=0, gap_days=1, validation_days=2))
    monkeypatch.setattr("src.experiment.safety_gate", lambda: {"passed": True})
    first = run_experiment(source, config, tmp_path / "first", cache_root=tmp_path / "cache")
    day = source.day

    def deny_training_read(date):
        # Date 3 may still be read as the initial API lag; earlier training dates cannot.
        assert date >= 3, "cache hit recomputed training preprocessing"
        return day(date)

    source.day = deny_training_read
    second = run_experiment(source, config, tmp_path / "second", cache_root=tmp_path / "cache")
    assert first["pooled_score"] == second["pooled_score"]
    record = json.loads((tmp_path / "second/fold_0/preparation_cache.json").read_text())
    assert record["hit"]
