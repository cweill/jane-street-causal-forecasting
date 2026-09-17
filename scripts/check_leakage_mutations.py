"""Demonstrate that causal tests reject deliberately leaky implementations.

Runs each mutation in a disposable copy; never modifies the working source files.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUTATIONS = [
    {
        "name": "expose_current_responder_in_test_frame",
        "file": "src/data/api_simulator.py",
        "replacements": [
            (
                "test.select(TEST_COLUMNS).clone()",
                'test.with_columns(pl.Series("responder_6", truth)).clone()',
            )
        ],
        "tests": [
            "tests/test_lag_visibility.py::test_api_lags_are_entire_previous_day_stamped_current_day"
        ],
    },
    {
        "name": "release_same_day_responders_early",
        "file": "src/data/api_simulator.py",
        "replacements": [
            (
                'if previous.height and previous["date_id"].unique().to_list() != [current_date - 1]:',
                "if False:",
            ),
            (
                "previous_day_lags(self._source.day(date - 1), date)",
                "previous_day_lags(self._source.day(date), date)",
            ),
        ],
        "tests": [
            "tests/test_no_future_leakage.py::test_mutating_all_unreleased_responders_cannot_change_predictions"
        ],
    },
    {
        "name": "fit_on_future_partition",
        "file": "src/training/offline.py",
        "replacements": [("dates = tuple(training_dates)", "dates = tuple(source.dates())")],
        "tests": [
            "tests/test_offline_boundaries.py::test_source_is_never_read_outside_training_partition"
        ],
    },
    {
        "name": "patrick_mixes_future_features_into_prefix",
        "file": "src/models/patrick_yam.py",
        "replacements": [
            (
                "x = torch.where(mask[..., None], x, 0.0).clamp(-10, 10)",
                "x = torch.where(mask[..., None], x, 0.0).clamp(-10, 10)\n        x = x + x.mean(dim=1, keepdim=True)",
            )
        ],
        "tests": [
            "tests/test_patrick_model.py::test_future_inputs_cannot_change_prefix_or_receive_prefix_gradient"
        ],
    },
    {
        "name": "patrick_fits_future_partition",
        "file": "src/training/patrick.py",
        "replacements": [("dates = tuple(dates)", "dates = tuple(source.dates())")],
        "tests": [
            "tests/test_patrick_pipeline.py::test_preprocessing_fits_only_training_and_unknown_categories_stay_unknown"
        ],
    },
]


def main():
    reports = []
    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory() as temp:
            copy_root = Path(temp)
            for directory in ("src", "tests", "scripts", "experiments", "configs"):
                shutil.copytree(
                    ROOT / directory,
                    copy_root / directory,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            shutil.copy(ROOT / "pyproject.toml", copy_root / "pyproject.toml")
            path = copy_root / mutation["file"]
            content = path.read_text()
            for old, new in mutation["replacements"]:
                if content.count(old) != 1:
                    raise RuntimeError(f"mutation anchor changed: {old}")
                content = content.replace(old, new)
            path.write_text(content)
            env = {**os.environ, "PYTHONPATH": str(copy_root), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *mutation["tests"]],
                cwd=copy_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            # Return code 1 indicates assertion/test failure, not an import/collection error.
            detected = result.returncode == 1 and "FAILED" in result.stdout
            reports.append(
                {"mutation": mutation["name"], "detected": detected, "test_output": result.stdout}
            )
            if not detected:
                raise AssertionError(
                    f"Mutation was not detected:\n{result.stdout}\n{result.stderr}"
                )
    output = ROOT / "artifacts/leakage-mutations.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Detected all {len(reports)} deliberate leakage mutations; report: {output}")


if __name__ == "__main__":
    main()
