import subprocess

import pytest
from test_patrick_pipeline import tiny_config

from src.config import load_config
from src.experiment import run_experiment


def differing_paths(a, b, prefix=""):
    paths = []
    for key in a:
        if isinstance(a[key], dict):
            paths += differing_paths(a[key], b[key], prefix + key + ".")
        elif a[key] != b[key]:
            paths.append(prefix + key)
    return paths


def test_gate_failure_stops_experiment_before_reading_data(monkeypatch, tmp_path):
    def failure(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "causal check failed", "")

    monkeypatch.setattr("src.safety.subprocess.run", failure)
    with pytest.raises(RuntimeError, match="experiments blocked"):
        run_experiment(None, tiny_config(), tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


def test_malformed_or_unknown_config_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("features:\n  future_targets: true\n")
    with pytest.raises(TypeError):
        load_config(path)
    path.write_text('online:\n  enabled: "false"\n')
    with pytest.raises(ValueError, match="booleans"):
        load_config(path)
