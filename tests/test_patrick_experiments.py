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
