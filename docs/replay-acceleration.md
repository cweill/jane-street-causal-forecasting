# Parallel replay from the existing Patrick checkpoint

The sequential run remains the reference implementation. The independent replay
runner reuses its immutable trained checkpoint and runs frozen and online replay
on separate L4 workers. There is no retraining, hyperparameter change, or change
to the date splits, label release, online update count, learning rate, or score.
Within each worker, days and API timestamps remain strictly chronological.

## Active run

[W&B dashboard](https://wandb.ai/cweill-self/janestreet-repro/runs/parallel-20260918T215904Z-c79fc1)
and [launch/verification record](references/patrick-parallel-replay-launch.json).

Launched from commit `10be09c`. The frozen worker imported 110 completed dates
(1380–1489). Its newly computed predictions on 1490–1491 exactly matched the
original worker; online predictions on 1380–1382 exactly matched the GPU benchmark.
Both replacements and their observer were verified running before the original
GPU and observer were intentionally stopped at 22:03 UTC on 2026-09-18.
Original checkpoints and results remain on the source volume. No training was
repeated. The original W&B run is labeled superseded and links to the replacement.

## Verified implementation changes

- Streaming `forward_step` constructs its own all-present mask for the current
  timestamp and avoids eight redundant GPU-to-host mask decisions. Missing symbols
  still retain their hidden states in the predictor's external symbol bank.
- A two-day evaluator-owned LRU cache reuses Parquet days and date metadata. The
  predictor still receives only the current public batch and released lags.
- Evaluator truth is partitioned by timestamp once per day instead of filtering
  the day's frame on each call. No truth cache is passed to the predictor.
- Each worker commits storage once per completed day, retaining all daily restart
  checkpoints. Its output directory is exclusive to that replay mode.
- Completed frozen days can be imported from the source run after verifying the
  initial checkpoint, split, source signatures, and latest frozen model weights.
  Predictions and statistics are copied unchanged. The last imported checkpoint
  includes complete restart state; earlier imported markers point to the source
  archive. Prefix import completes before replay resumes.

The original source volume is mounted read-only. Accelerated results use the
separate `janestreet-replay-acceleration` volume. An interrupted worker resumes
from the last committed day using the same code and launch identity.

## L4 benchmark

[Recorded evidence](references/patrick-replay-acceleration.json) uses the actual
five-epoch initial checkpoint on dates 1380–1382:

| Replay | Original | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Frozen | 59.50 s | 52.84 s | 1.126× |
| Online | 59.93 s | 58.86 s | 1.018× |

These are three-day measurements, not a stable throughput guarantee. Online speed
improvement is small enough that timing noise may explain it. Parallel execution
is the main expected wall-time reduction. Prediction values, scores, final model
weights, and online Adam state matched exactly. Perturbing the last day's future
responders changed neither predictions nor online weights/optimizer state.

The new independent runner is also compared against the sequential reference in
tests, including online interruption/resume, frozen-prefix reuse, unchanged row
IDs, and rejection of mismatched comparison checkpoints.

## Launch and monitoring

```bash
TMPDIR=/tmp uv run --with modal==1.5.5 modal deploy scripts/modal_parallel_replay.py
TMPDIR=/tmp uv run --with modal==1.5.5 python -m scripts.modal_parallel_replay \
  --source-run 20260918T200909Z-2b3a45f0 \
  --benchmark-id benchmark-20260918T214616Z \
  --benchmark-commit 0a1dd1c
```

Submission requires a clean commit, verifies the benchmark's complete source
fingerprint against that Git commit, and rejects changes to the benchmarked kernels.
The setup job runs the causal tests again. Both GPU calls and the CPU observer are
spawned independently; their IDs are saved locally and on the output volume.
No long-lived parent call owns the workers.

The observer creates a new W&B run grouped under the original training run. It
uses separate `offline/date_id` and `online/date_id` axes, so whichever worker
finishes a day first cannot suppress the other's later metrics. It uploads the
combined blue/orange rolling-20-day plot when both jobs finish. W&B open log files
live on a separate tracking volume so result-volume reloads remain safe.

When migrating an active evaluation, leave the original job running until the new
jobs have committed results and the observer has reported both modes. Then stop
the superseded worker and its observer to avoid paying for duplicate evaluation.
Retain all source artifacts and record the replacement job/run IDs.
