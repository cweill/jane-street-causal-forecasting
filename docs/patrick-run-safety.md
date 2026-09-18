# Patrick online-learning plot: run and recovery

The run is a reconstruction, not a claim of matching Patrick's exact scores. All
choices are frozen in `configs/patrick_ol.yaml`: one seed (0), five fixed epochs,
full published network dimensions, offline dates 0–1379, replay 1380–1698, and
primary scoring 1500–1698. Five epochs is a research budget. Online Adam uses
three daily steps, betas (0.8, 0.95), and **lr=5e-4**, the user's hypothesis.
The first replay day has visible lags but no cached prior-day inputs; the first
update therefore occurs at 1381. This source ambiguity remains explicit.

The first detached full run was launched as `20260918T003149Z-1720d9aa` from
commit `82c02ad`. Its [launch manifest](references/patrick-ol-launch.json) records
the immutable input/configuration and [Modal App](https://modal.com/apps/cweill/main/ap-DTPNnpaStYiJFcJg7cpifn).
It was cancelled at 17:44 Pacific on September 17 before a training checkpoint
was written. CPU preparation completed and its cache survived. The local launcher
reported a network error; the detached invocation did not provide the intended
lifetime isolation. See the [incident evidence](references/patrick-ol-interruption.json).
The corrected launcher uses a deployed App and `.spawn()`, saving a job ID and
exiting immediately. A [real submission-lifetime test](references/patrick-submission-rehearsal.json)
proved that its CPU probe remained pending after the submitter exited, then completed.

The corrected run is `20260918T200909Z-2b3a45f0`, submitted from commit `c4dcde6`
with FunctionCall `fc-01M2V26FSA5FZHV9AR3E7JRP8A`. Its
[relaunch manifest](references/patrick-ol-relaunch.json) preserves the unchanged
model/data settings and links the persistent deployment. The original run remains
available as failure evidence; the compatible preparation cache is reused.

Later, Modal preempted that coordinator. Its restarted preparation call hit a
tuple-versus-list identity-comparison bug, but the independent GPU call
`fc-01M2V292F49YDZ1KSKJ2NS6M0J` continued training. Monitoring was redirected to
that GPU call. The [incident record](references/patrick-coordinator-preemption.json)
documents the correction; the active trainer was not redeployed. Future launch
comparison uses canonical JSON and still rejects actual source/config/data changes.
Do not rerun the failed coordinator while its GPU child is active.

Both curves start from the same checkpoint. The diagnostic plot includes the
120-day warmup and uses pooled weighted zero-mean R² over complete trailing
20-day windows. Primary scores exclude warmup. These replay dates are being
used for the requested reconstruction, so they must not subsequently be treated
as an untouched holdout for tuning.

## Evidence before the long run

- [Original CUDA pilot](patrick-pilot-report.md): full network dimensions, real
  data, delayed labels, independent metric checks, and future-label perturbations.
- [Cache and inference benchmark](references/patrick-cache-benchmark.json): on
  one L4, cached preprocessing and recurrent-state indexing produced bit-identical
  prepared arrays, predictions, and final weights relative to the original pilot.
  Four-day cold preparation took 1.440 seconds; a verified cache hit took 0.150.
  Frozen five-day replay fell from 137.716 to 112.074 seconds (18.6% less time),
  and online replay from 151.786 to 111.375 seconds (26.6% less time).
  This comparison retained the old pilot's online lr=3e-4 to isolate code changes.
- [CUDA recovery rehearsal](references/patrick-resume-rehearsal.json): interrupted
  two-epoch training resumed with exactly identical weights and loss history,
  including a dropout=0.1 RNG stress test. Interrupted online replay at lr=5e-4
  also produced exactly identical predictions and weights. The rehearsal took
  108.19 seconds on one L4 and passed its 76-test remote causal gate.
- Local paired-run tests reproduce uninterrupted results after a day-boundary
  interruption. NaN loss and NaN gradient tests ensure failure occurs before an
  unhealthy optimizer step overwrites the last healthy training checkpoint.
- All five deliberate leakage mutations were detected by failing tests.

These are small real-data rehearsals, not proof of full-period forecast quality.
The earlier 7.5-hour estimate used the slower pilot timings. New measurements
suggest roughly six hours, but full-data parquet I/O, checkpoint commits, startup,
and early-day sequence lengths remain unmeasured at scale. The timeout is a
failure bound, not an estimate or target runtime.

## Persistent cache and checkpoints

`src/training/cache.py` keys preprocessing by actual parquet contents, training
dates, feature settings, relevant preprocessing code, and NumPy/Polars versions.
Cache hits verify file checksums. Seeds and learning rates can reuse a compatible
cache; changes to training boundaries or preprocessing cannot. Fit statistics and
category vocabularies use only offline training dates.

`src/training/checkpoints.py` saves model and optimizer state, RNG states, epoch,
shuffled date order and position. Training snapshots are atomic every 25 daily
batches and at each epoch boundary. Online replay snapshots include online Adam
moments, the previous day's public input cache, update history and row offset;
they are saved after every complete day. Prediction outputs and daily scoring
statistics are also retained. Restart identity checks reject different code,
data, model settings, seeds, or training configurations.

Snapshot files are explicitly committed to a persistent Modal Volume. An abrupt
failure can lose work since the last successful commit; recovery continues from
that checkpoint rather than assuming the latest progress message is durable.
The Volume retains older daily replay checkpoints for audit/recovery. No automatic
retry loop is enabled, so deterministic failures stop and retain their evidence.

## Launch and monitor

From a clean, committed checkout with Modal credentials already configured:

```bash
TMPDIR=/tmp uv run --with modal==1.5.5 \
  modal deploy scripts/modal_patrick_reproduction.py
TMPDIR=/tmp uv run --with modal==1.5.5 \
  python -m scripts.modal_patrick_reproduction
```

The launcher runs the causal gate locally and remotely, verifies the fixed date
coverage and SHA256 identity of all ten data partitions, and uploads only explicit
code paths and competition training files. No Kaggle credentials are uploaded.
CPU preparation completes and commits its cache before an L4 is allocated.
The CPU stage is capped at two hours, the single GPU stage at twelve hours, with
no retries. Both stages run under a server-side coordinator. The launcher follows
Modal's [deployed job submission pattern](https://modal.com/docs/guide/job-queue):
`.spawn()` queues work and returns a FunctionCall ID. The submitting process exits
without awaiting training or maintaining an ephemeral App context. Volume commits
persist recovery state; polling is independent of the submitting process.
Do not redeploy the App while a run is active.

The command prints the run ID and saves `submission-*.json` with a FunctionCall ID. The persistent Volume is
`janestreet-patrick-reproduction`:

- `/launches/<run-id>/`: source archive, exact commit/config/data manifest, splits,
  causal gate result, and CPU preparation report.
- `/preparation_cache/<key>/`: reusable, verified training arrays and fitted state.
- `/runs/<run-id>/status.json`: latest phase, training batch, or replay date.
- `/runs/<run-id>/training.pt`: training recovery state.
- `/runs/<run-id>/{offline,online}/checkpoints/date_<date>/`: daily replay recovery.
- `/runs/<run-id>/online_learning.{png,svg,csv}`: completed curves and numerical data.
- `/runs/<run-id>/result.json`: completed primary scores and update audit.
- `/runs/<run-id>.tar.gz`: compact result archive; large recovery state stays on Volume.

Poll with `modal.FunctionCall.from_id(call_id).get(timeout=0)` in a separate
process; `TimeoutError` means pending. After successful completion, use `modal volume get janestreet-patrick-reproduction
/runs/<run-id>.tar.gz artifacts/` after completion and verify against
`/runs/<run-id>/download.json`. Never interpret `status.json` alone as a completed
result; require `result.json` and the plot artifacts.

To resume, first confirm the original FunctionCall has terminated to avoid two writers.
A deployed App can remain idle after its call finishes. Restore
the recorded commit (or use its source archive), then repeat the launch command
with `--run-id <same-id>`. Resume requires the same source, configuration and input
identity, and reuses validated preparation and model checkpoints. A code change
requires a new run; do not bypass the identity guard.

This launcher does not select epochs, change learning rates, or add seeds based
on the plotted results. Any later method experiment must be a separately named run.
