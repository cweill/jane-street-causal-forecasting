# W&B monitoring without restarting training

The **current evaluation** runs frozen and online replay concurrently:

https://wandb.ai/cweill-self/janestreet-repro/runs/parallel-20260918T215904Z-c79fc1

See [parallel replay](replay-acceleration.md) and the
[verified handoff](references/patrick-parallel-replay-launch.json). It reuses the
original trained checkpoint and 110 completed frozen days. The original worker
and observer were stopped intentionally after checking replacement predictions.

The **archived training run** and original sequential-observer design below are at:

https://wandb.ai/cweill-self/janestreet-repro/runs/20260918T200909Z-2b3a45f0

[Connection evidence](references/patrick-wandb-monitor.json) records the deployed
observer's job ID and metrics independently read back from the W&B API.

The `janestreet-repro` project was created with private visibility in the user's
`cweill-self` entity. The monitor uses the existing Modal `wandb` secret. Credentials
are never added to code or run configuration.

## What is logged

- Training batch count, fractional epoch, progress, checkpoint optimization losses,
  and completed-epoch mean optimization losses.
- Newly instrumented runs also save unbalanced multitask loss and responder_6
  training R², with batch and epoch charts (definitions below).
- Separate offline/online daily diagnostic R², complete trailing 20-day pooled R²,
  and cumulative R² over scored dates only. Warmup is explicitly marked and excluded
  from scored cumulative metrics. No validation pass is added during training.
- Final offline/online scores and the generated comparison plot after both replays.
- Frozen model configuration, original code/data fingerprints, source job ID,
  last observation time, and source phase.

**Optimization loss is not a prediction-error curve.** Patrick's detached loss
balancing divides each target loss by its own detached value; the displayed value
mainly reflects sample weighting. Use weighted zero-mean R² to assess predictions.
Raw unbalanced loss was not saved by the original trainer and cannot be recovered
from its stored loss values. The monitor does not fabricate it or tune the run.

### Training diagnostics for subsequent runs

The trainer now saves these additional scalar observations from its existing
**pre-update, train-mode forward pass**, including dropout. It performs no extra
model forward pass, validation pass, or parameter update.

| W&B metric | Definition |
| --- | --- |
| `train/unbalanced_loss` | Target-weighted average of each active target's weighted SSE / target energy, before detached balancing and date multipliers. Respects the auxiliary-target switch. |
| `train/responder_6_r2` | Daily weighted zero-mean R² for responder_6, using competition row weights. |
| `train/epoch_mean_unbalanced_loss` | Arithmetic mean of defined daily unbalanced losses across the epoch. |
| `train/epoch_responder_6_r2` | One minus pooled responder_6 SSE divided by pooled target energy across the epoch; not the mean of daily R². |

Diagnostics use float64 reductions and exclude zero-weight rows. Zero-energy R²
and losses with no active target energy are undefined and omitted from W&B.
Training R² includes predictions from successive model states during the epoch;
it is **not held-out R² or an evaluation of the final epoch checkpoint**. It does
not replace causal replay or select epochs. Training date multipliers affect
optimization only, not these diagnostics.

Current-epoch diagnostic records are saved alongside losses in atomic training
checkpoints, restored on resume, and pooled into persistent epoch summaries. The
monitor accepts older checkpoints without diagnostics and leaves those charts
absent. Run `20260918T200909Z-2b3a45f0` completed training on its original image;
neither its trainer nor its observer was restarted for this logging change. Its
evaluation was later superseded by the parallel workers linked above. New trainer
and monitor deployments expose the additional metrics on future training runs.

Verification: a three-epoch CPU fixture with dropout produced exactly the same
model tensors and original optimization losses before and after instrumentation.
Unbalanced loss changed from 1.282719 to 1.252521 while normalized loss stayed
constant. Tests also cover hand-calculated errors, target selection, undefined
scores, pooled R², and identical diagnostic histories after interrupted training.

Individual loss points are recovered from each observed checkpoint's current epoch;
completed-epoch means remain available afterward. If a polling gap crosses an epoch
boundary, some individual batch losses may be unavailable. Points retain their actual
batch numbers; missing observations are not interpolated. Epoch summaries still appear.

## Isolation and recovery

`scripts/modal_wandb_monitor.py` deploys a separate CPU-only App with a **read-only**
mount of the research Volume. It does not import the trainer or predictor, deploy the
training App, update model weights, or send metrics back to training. Only scalar
metrics, configuration/provenance, and the final plot are sent to W&B. Model weights,
raw data, features, labels, row-level predictions and credentials are not uploaded.
Observer CPU system metrics are disabled because they would misrepresent trainer
GPU utilization.

The observer polls every 60 seconds, runs with 0.25 CPU and 2 GiB RAM, and has a
12-hour execution cap with no retries. Its logs and heartbeat live on the separate
`janestreet-wandb-monitor` Volume under the source run ID. A tracker failure cannot
cancel the independent training job. Use the source FunctionCall and Modal run
artifacts to distinguish a training failure from an observer failure.

The W&B run ID equals the source run ID. Metrics have deterministic ordered steps,
with separate batch/epoch/date chart axes. On observer restart, W&B restores the
next step and already-logged observations are skipped. Do not start two observers
for the same W&B run simultaneously.

## Archived sequential observer launch

Deploy only this monitor App; do not redeploy the active training App:

```bash
TMPDIR=/tmp uv run --with modal==1.5.5 \
  modal deploy scripts/modal_wandb_monitor.py
TMPDIR=/tmp uv run --with modal==1.5.5 python -m scripts.modal_wandb_monitor \
  --run-id 20260918T200909Z-2b3a45f0 \
  --source-call-id fc-01M2V292F49YDZ1KSKJ2NS6M0J
```

The submission exits immediately and saves the observer FunctionCall ID and W&B URL
in `artifacts/patrick-reproduction/<run-id>/wandb-monitor.json`. W&B 0.30.0 is pinned
in the observer image; the trainer's lockfile and running image are unchanged.

The original observer watched the **GPU FunctionCall directly**. The original
coordinator was preempted and failed on restart while its spawned GPU child kept
training; a parent-watching observer consequently reported failure. See the
[preemption record](references/patrick-coordinator-preemption.json). A coordinator's
failure is not proof that its separately spawned GPU child stopped. Check the
child call and saved checkpoints before restarting training or creating another writer.

Tests cover metric payloads excluding checkpoint tensors, true batch positions,
undefined/invalid values, committed-day visibility, warmup exclusion, pooled rolling
scores, and ordered offline/online steps. Core training and causal tests also pass.
