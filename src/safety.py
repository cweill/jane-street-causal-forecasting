"""Mandatory experiment entry-point gate; no leaderboard tuning before causal checks."""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAUSAL_TESTS = [
    "test_metric.py",
    "test_lag_visibility.py",
    "test_no_future_leakage.py",
    "test_features.py",
    "test_model.py",
    "test_offline_boundaries.py",
    "test_online_update_timing.py",
    "test_gateway_parity.py",
    "test_patrick_model.py",
    "test_patrick_pipeline.py",
    "test_pilot_boundaries.py",
    "test_fixed_protocols.py",
    "test_preparation_cache.py",
    "test_rolling_plot.py",
    "test_resume.py",
    "test_replay_acceleration.py",
    "test_parallel_replay.py",
]


def safety_gate():
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *[str(ROOT / "tests" / t) for t in CAUSAL_TESTS],
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "causal/protocol gate failed; experiments blocked\n" + result.stdout + result.stderr
        )
    files = sorted(
        [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("tests/**/*.py"),
            *ROOT.glob("scripts/**/*.py"),
            *ROOT.glob("experiments/**/*.py"),
            *ROOT.glob("tests/fixtures/*.json"),
            *ROOT.glob("configs/*.yaml"),
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
        ]
    )
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(ROOT)).encode())
        digest.update(file.read_bytes())
    return {
        "passed": True,
        "code_and_tests_sha256": digest.hexdigest(),
        "command": command,
        "output": result.stdout,
    }
