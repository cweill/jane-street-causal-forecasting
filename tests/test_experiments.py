import copy
import json
import subprocess
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from experiments.ablations import ablation_configs
from src.artifacts import load_predictor, save_predictor
from src.config import ExperimentConfig, load_config
from src.data.api_simulator import APISimulator
from src.data.features import FeatureConfig
from src.data.loader import FrameSource, ParquetSource
from src.experiment import run_experiment
from src.models.grigoreva_gru import ModelConfig
from src.training.offline import prepare_training, train_model
from src.training.online import OnlineConfig, StreamingPredictor


def differing_paths(a, b, prefix=""):
    paths = []
    for key in a:
        if isinstance(a[key], dict):
            paths += differing_paths(a[key], b[key], prefix + key + ".")
        elif a[key] != b[key]:
            paths.append(prefix + key)
    return paths


def test_ablation_changes_exactly_one_switch_and_seed_count():
    config = load_config("configs/online.yaml")
    variants = list(ablation_configs(config))
    assert len(config.members) == 6
    assert len(variants) == 6
    for _, variant in variants[1:]:
        paths = differing_paths(config.to_dict(), variant.to_dict())
        assert len([p for p in paths if p != "name"]) == 1
    assert len(variants[-1][1].members) == 2  # architecture ensemble is unchanged


def test_gate_failure_stops_experiment_before_reading_data(monkeypatch, tmp_path):
    def failure(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "causal check failed", "")

    monkeypatch.setattr("src.safety.subprocess.run", failure)
    with pytest.raises(RuntimeError, match="experiments blocked"):
        run_experiment(None, ExperimentConfig(), tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


def test_parquet_source_and_checkpoint_roundtrip_match_stream(panel, tmp_path):
    path = tmp_path / "panel.parquet"
    panel.write_parquet(path)
    source = ParquetSource(path)
    assert source.dates() == FrameSource(panel).dates()
    assert source.day(2).equals(FrameSource(panel).day(2))
    prepared = prepare_training(source, (0, 1, 2), FeatureConfig(True, True, 3), tmp_path / "train")
    model, _ = train_model(
        prepared, ModelConfig(hidden_sizes=(4,), linear_sizes=(), dropout=(0.0,), linear_dropout=())
    )
    save_predictor(
        tmp_path / "model", [model], prepared.features, prepared.scaler, OnlineConfig(True), [0]
    )
    fresh = StreamingPredictor(
        [copy.deepcopy(model)],
        copy.deepcopy(prepared.features),
        prepared.scaler,
        OnlineConfig(True),
        seeds=[0],
    )
    restored = load_predictor(tmp_path / "model")
    a = APISimulator(source, [3, 4]).run(fresh.predict).predictions["responder_6"]
    b = APISimulator(source, [3, 4]).run(restored.predict).predictions["responder_6"]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    # Competition deployment can rebase test dates to zero while preserving history.
    rebased = source.day(3).with_columns(pl.lit(0).alias("date_id"))
    deploy = load_predictor(tmp_path / "model", reset_clock=True)
    APISimulator(FrameSource(rebased), [0]).run(deploy.predict)


def test_runner_writes_splits_scores_checkpoint_and_update_audit(panel, tmp_path, monkeypatch):
    # The gate itself is tested above and exercised by CLI smoke runs; avoid nested pytest.
    monkeypatch.setattr(
        "src.experiment.safety_gate", lambda: {"passed": True, "test_fixture": True}
    )
    config = load_config("configs/baseline.yaml")
    config = replace(
        config,
        model=replace(
            config.model, hidden_sizes=(4,), linear_sizes=(), dropout=(0.0,), linear_dropout=()
        ),
        training=replace(config.training, epochs=1),
        online=OnlineConfig(True),
        cv=replace(config.cv, min_date=0, n_splits=1, validation_days=2, gap_days=1),
    )
    output = tmp_path / "run"
    result = run_experiment(FrameSource(panel), config, output)
    split = json.loads((output / "splits.json").read_text())[0]
    assert split["train_dates"] == [0, 1, 2, 3]
    assert split["warmup_dates"] == [4]
    assert split["validation_dates"] == [5, 6]
    assert result["folds"][0]["online_updates"] == 2
    assert result["folds"][0]["scored_rows"] == 20
    assert (output / "fold_0/checkpoint/weights.pt").exists()
    assert len(list((output / "fold_0/predictions").glob("*.parquet"))) == 3
    assert json.loads((output / "fold_0/updates.json").read_text())[0]["source_date"] == 4


def test_malformed_or_unknown_config_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("features:\n  future_targets: true\n")
    with pytest.raises(TypeError):
        load_config(path)
    path.write_text('features:\n  rolling: "false"\n')
    with pytest.raises(ValueError, match="booleans"):
        load_config(path)
