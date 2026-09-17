# Modal GPU correctness pilot

The first successful real-data run is recorded in the
[correctness and runtime report](real-data-pilot-report.md).

The pilot uses the existing Modal profile and a dedicated `janestreet-repro-pilot`
Volume. It launches an ephemeral App, not a scheduled or persistent deployment.
The uploaded dataset contains exactly dates 700–708; full competition data and
Kaggle credentials stay local. Code upload is limited to src, tests, scripts,
experiments, configs, pyproject.toml, and uv.lock.

## Run

From the repository root, prepare a new slice (this runs the causal gate):

```bash
.venv/bin/python -m scripts.real_data_pilot prepare \
  --output artifacts/pilot-input.parquet

TMPDIR=/tmp uv run --no-project --python 3.12 --with modal==1.5.5 \
  modal run scripts/modal_pilot.py --data artifacts/pilot-input.parquet

# The same workload and checks, using Patrick’s reconstruction:
TMPDIR=/tmp uv run --no-project --python 3.12 --with modal==1.5.5 \
  modal run scripts/modal_pilot.py --data artifacts/pilot-input.parquet --method patrick
```

The Modal SDK is installed in an isolated uv environment. The remote image uses
Python 3.12 and the project's frozen uv.lock, including test dependencies. The
project code and test fixtures are mounted explicitly; the local .venv is not used
on the GPU machine. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before PyTorch starts.

The same pilot can run locally:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.real_data_pilot run \
  --data artifacts/pilot-input.parquet --output artifacts/cpu-pilot --device cpu
```

## Fixed workload and limits

- One L4 GPU; one container; no automatic application retries.
- Four CPU cores and 16 GiB host memory, each with an explicit limit.
- 20-minute function execution timeout and separate 15-minute startup timeout.
- One seed and one epoch for either method.
- Grigoreva: published gru3 widths 250/150/150, four auxiliary branches,
  full 125-input pipeline with market averages and rolling window 1000.
- Patrick: eight blocks, width 64, eight attention heads, GRU width 256,
  77 numerical inputs and three categorical embeddings, all nine targets.
- Training dates 700–703; unscored warmup 704–705; scoring 706–708.
- Three sequential replays: frozen model, online updates, and online updates with
  all nine responders on date 708 changed to test unreleased-label invariance.

At the [Modal rates inspected on 2026-09-17](https://modal.com/pricing), L4 costs
$0.000222/second, CPU $0.0000131/core/second, and memory $0.00000222/GiB/second.
Twenty minutes at the listed requested compute resources is approximately $0.37.
This is an estimate, not an account spending cap: image builds, startup, storage,
and platform billing details are separate. No larger training job is scheduled.

## Checks and artifacts

The safety gate must pass remotely before training. The pilot then checks the
exact date set, lag keys and values against the previous historical day, absence
of intraday lag releases, update source dates and row counts, unchanged fitted
normalization, and exclusion of warmup from the aggregate metric. It compares
initial model hashes across off/on runs and requires equal predictions before
the first online update.

Changing date 708 responders must leave every replay prediction and final model
weight unchanged: those labels cannot be released before date 709. The altered
truth is still evaluator-owned, so its score is intentionally not a model-quality
comparison. The normal short-pilot scores also have no leaderboard interpretation.

The first replay lag carries date 703 labels, but the fresh predictor has no cached
replay inputs for that day and skips updating on them. Expected online updates are
at time zero on dates 705, 706, 707, and 708, using dates 704–707 respectively.

The remote Volume retains inputs under `/inputs/<sha256>.parquet`, and run outputs
under `/runs/<run-id>`. Successful runs are archived and downloaded automatically
to `artifacts/modal-pilot/<run-id>/`, with SHA256 verification before extraction.
Checkpoints, prepared training arrays, predictions, update/lag audits, configuration,
data inventory, source hash, timings, host peak RSS, and CUDA peak allocated memory
are retained. Failed calls attempt to commit partial artifacts in `finally`;
hard termination/preemption does not guarantee a final commit.

No credentials are embedded in scripts. If interrupted, the Modal App URL and
Volume remain the places to inspect status and recover artifacts; do not launch
another job without checking whether the original call is still active.
