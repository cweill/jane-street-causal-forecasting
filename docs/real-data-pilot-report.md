# Real-data correctness pilot: Modal L4

Completed 2026-09-17. **Passed.**

[Modal run](https://modal.com/apps/cweill/main/ap-2BCuNQ6ve5y9d4S1ofk3uO)
used the authenticated `cweill` profile. The App stopped after completion and the
subsequent status check showed zero running containers. No persistent deployment
or scheduled training was created.

Local artifacts: `artifacts/modal-pilot/20260917T225426Z-15801316/`.
Remote artifacts: Volume `janestreet-repro-pilot`,
`/runs/20260917T225426Z-15801316/`. The downloaded archive's SHA256 was verified.
See [run instructions and compute limits](modal-pilot.md).

## Workload

The slice contains 251,680 real rows (63.3 MiB parquet), dates 700–708 only.
Offline training used 700–703, warmup 704–705, and scoring 706–708. The held-out
1499–1698 responders were neither exported nor used for this pilot.

One seed and one epoch used published gru3 widths 250/150/150, all four auxiliary
branches, and the full 125-input feature pipeline, including rolling window 1000
and same-timestamp market averages. This is a small training interval, not reduced
model dimensions. Both normal replays loaded the identical saved checkpoint.

## Verification evidence

| Check | Result |
|---|---|
| Local full test suite | 49 passed |
| Local deliberate leakage mutations | All three detected before the pilot |
| Remote causal/protocol safety gate | 42 passed before training |
| Published dimensions with CUDA deterministic algorithms | Training and all replays completed |
| Exact input dates and upload integrity | Dates 700–708; SHA256 checked remotely |
| Lag visibility and keyed values | All five day-start releases matched day d−1; no intraday releases |
| Online update timing | Four updates at (705,0) through (708,0), using days 704–707 |
| Fitted normalization | Mean and scale unchanged across replay |
| Matched starting weights | Same hash for online off/on |
| Frozen replay weights | Initial and final hashes identical |
| First warmup day predictions | Online off/on identical, before any eligible update |
| Future responder perturbation | All nine date-708 responders changed; every replay prediction and final weights identical |
| Scored row count | 84,216; warmup excluded |
| Independent metric recomputation | Keyed predictions/truth join and float64 NumPy formula agree within 1e-12 |
| Local lint and formatting | Passed |

Each replay made 4,840 API-style calls (968 timestamps × five days). Three replays
completed: online off, online on, and online on with unreleased date-708 responders
perturbed. The last test changes evaluator truth; its score is not a model comparison.

The first replay lag supplies labels for date 703. The fresh predictor has no
cached replay inputs for that date, so its first update occurs at date 705. This
matches the existing harness behavior; it does not resolve Patrick Yam's policy
for reusing the final offline day.

The training panel naturally has 28 symbols on date 702 and 29 on the other dates.
The replay panel has 29 symbols each day. Missing/new-symbol edge cases beyond
those observed in this slice remain covered by synthetic tests; this run is not
evidence for every possible changing-symbol pattern.

## Runtime and memory

Measured on one NVIDIA L4, four CPU cores, and a 16-GiB host-memory allocation.
Software: Python 3.12, torch 2.14.0, polars 1.44.2, numpy 2.5.3, frozen uv.lock.

| Measurement | Value |
|---|---:|
| Training feature preparation | 10.56 s |
| Offline training, four daily batches / one epoch | 3.90 s |
| Frozen replay, five full days | 98.97 s |
| Online replay, five full days | 103.36 s |
| Complete measured pilot work, including third replay and checks | 320.70 s |
| Host process peak RSS | 5.12 GiB |
| Peak PyTorch CUDA allocated memory | 2.09 GiB |
| Frozen prediction-call median / p95 / maximum | 17.49 / 19.47 / 26.99 ms |
| Online prediction-call median / p95 / maximum | 18.03 / 20.02 / 489.18 ms |

Prediction-call timings synchronize CUDA before and after each call; they include
feature construction and any online update inside the predictor. The complete-work
timer excludes the remote safety gate, image build, startup, upload, and archive
download. GPU allocated memory is PyTorch's measurement, not total device occupancy.
Actual account charges were not retrieved.

Training was a small part of this pilot's runtime; sequential timestamp replay
dominated. These timings do not establish a CPU/GPU speedup because no matching CPU
benchmark was run. They also do not extrapolate directly to Patrick's architecture
or a large seed ensemble.

## Diagnostic scores only

| Run | Weighted zero-mean R² |
|---|---:|
| Online off | -0.04798892936 |
| Online on | -0.01368802773 |

Both scores are below the zero predictor. Four training dates and one epoch were
chosen to test execution and causality, not predictive quality. This is not a
reproduction of either author's reported score or evidence that online learning
will improve a properly trained model on the final holdout.

## Artifacts and execution notes

The run directory contains `result.json`, `independent_verification.json`,
`config.json`, `splits.json`, `data_manifest.json`, `safety_gate.json`, training
history/cache, the initial checkpoint, and predictions and lag/update logs for
each replay. The source checksum in `safety_gate.json` identifies the executed
source and tests.

The first Modal attempt failed before training because the launch module was not
on the container Python path. That App was stopped, `/project/scripts` was added
to PYTHONPATH, and the bounded pilot was rerun successfully. Both Apps were confirmed
stopped afterward. No ongoing GPU job remains from this task.
