# Jane Street 2025 forecasting research harness

This package implements the Jane Street Real-Time Market Data Forecasting evaluation
information flow. Current research focuses on Patrick Yam's second-place method;
an independent reproduction of Evgeniia Grigoreva's eighth-place GRU pipeline is also
implemented. **Correctness comes before experiments:** every CLI training/ablation run
first executes the metric, protocol, and causal tests. No leaderboard optimization has
been performed, and no leaderboard score reproduction is claimed.

Patrick Yam’s second-place architecture is implemented as a documented
[reconstruction](docs/patrick-implementation.md), with train-only preprocessing,
causal asset attention/GRU inference, and delayed online updates. The
[reproduction tracker](docs/patrick-yam-tracker.md) preserves the source screenshots,
reported results, and unresolved details. Both methods use the same simulator and metric.

## Current Patrick results

The working baseline is **five offline epochs and online LR `1e-4`**, with three
daily Adam steps, betas `(0.8, 0.95)`, persistent online optimizer state, and all nine
responder targets. Offline training LR remains `5e-4`. Online updates use only
previous-day labels after their API release.

Completed experiments report pooled weighted zero-mean R²:

| Experiment | Seeds | Scored dates | Frozen R² | Online R² at `1e-4` |
|---|---:|---|---:|---:|
| Development, 3 offline epochs | 3 | 1180–1379 | 0.01818019 | 0.02193983 |
| Development, 4 offline epochs | 3 | 1180–1379 | 0.01893456 | 0.02308571 |
| Development, 5 offline epochs | 3 | 1180–1379 | 0.01935094 | 0.02405157 |
| Later follow-up, 5 offline epochs | 17 | 1500–1698 | 0.01412888 | 0.01941540 |

Development models train on dates 0–1059 and replay unscored warmup 1060–1179.
The 17-model ensemble trains on 0–1379 and replays unscored warmup 1380–1499.
Scores from these different windows are not directly comparable.

The [completed epoch study](docs/patrick-epoch-study.md) favors five epochs among
the three tested budgets. The [learning-rate refinement](docs/patrick-online-refinement.md)
found `5e-5` narrowly ahead of `1e-4` on the development window (0.02406497 versus
0.02405157); this does not establish a precise optimum. Only `1e-4` has the
[completed 17-model follow-up](docs/patrick-online-followup.md), where it improves
R² over frozen by 0.00528652. These windows have been inspected during development;
they are not untouched test sets, and Patrick's exact published scores are not reproduced.

Historical configs retain their original settings: for example,
`configs/patrick_ensemble.yaml` still has online LR `5e-4`. The follow-up launcher
overrides that value to `1e-4`; it does not retrain the ensemble. The epoch-study
config trains through epoch four to capture epochs three and four and reuses the
original epoch-five results. Use each linked runbook and recorded launch identity
to reproduce that experiment rather than treating every config as the current baseline.

The completed [W&B epoch comparison](https://wandb.ai/cweill-self/janestreet-repro/runs/epoch-study-20260920T185609Z-overview)
and [17-model OL comparison](https://wandb.ai/cweill-self/janestreet-repro/runs/ol-followup-20260920T070707Z)
contain the scores and rolling charts.

## Workflows and implementation

The [bounded Modal GPU pilot](docs/modal-pilot.md) runs the real-data correctness
checks on dates 700–708 and downloads its checkpoints, audits, and runtime report.
The [first L4 pilot passed](docs/real-data-pilot-report.md), including a real-data
future-responder perturbation check. The
[Patrick L4 pilot also passed](docs/patrick-pilot-report.md) on the identical slice.
The [Patrick plot runbook](docs/patrick-run-safety.md) records the cache benchmark,
CUDA interruption/recovery rehearsal, fixed run settings, and persistent Modal launcher.
The [W&B monitor](docs/wandb-monitoring.md) can observe a Patrick job's saved
losses and evaluations through a separate CPU process with read-only data access.
Patrick's optional [stacked ensemble inference](docs/stacked-ensemble-inference.md)
batches model weights and GRU states across seeds while retaining separate online optimizers.
The [scaling profile](docs/patrick-scaling-profile.md) separates timestamp inference,
daily updates, and preparation costs for 1, 4, and 17 models on an L4.
The [17-seed ensemble runbook](docs/patrick-ensemble-run.md) describes verified
seed-0 reuse, four concurrent training GPUs, recovery checkpoints, and automatic
matched frozen/online replay.
[Epoch-level validation](docs/patrick-epoch-validation.md)
logs held-out frozen-model R² to W&B and saves each epoch's checkpoint using an
earlier chronological split.
The [online sensitivity study](docs/patrick-online-sensitivity.md) compares three
learning rates and two Adam reset policies against a fixed three-seed frozen baseline
on an earlier development interval.
The [lower-rate refinement](docs/patrick-online-refinement.md) reuses those models
and baselines to evaluate four additional rates with scored time-block diagnostics.
The [epoch-budget study](docs/patrick-epoch-study.md) compares epochs three, four and
five with a fixed online learning rate, reusing the epoch-five results.

## Run locally

Python 3.12 is the tested runtime. From this checkout:

```bash
uv sync --python 3.12 --frozen --extra dev
uv run --frozen js-repro verify
uv run --frozen python -m pytest -q
uv run --frozen python -m scripts.check_leakage_mutations

# Small synthetic runs; reduced network dimensions, one epoch, no financial meaning.
uv run --frozen js-repro smoke --config configs/baseline.yaml --output artifacts/baseline-smoke
uv run --frozen js-repro smoke --config configs/online.yaml --output artifacts/online-smoke
uv run --frozen js-repro smoke --config configs/patrick.yaml --output artifacts/patrick-smoke

# Real research: supply the competition training parquet downloaded from Kaggle.
uv run --frozen js-repro run --data /path/to/train.parquet \
  --config configs/online.yaml --output artifacts/online-cv

# Reference plus five independent one-switch ablations, using identical folds/epochs.
uv run --frozen js-repro ablations --data /path/to/train.parquet \
  --config configs/online.yaml --output artifacts/ablations
```

`--data` accepts a parquet file or a directory containing parquet partitions. Keep the
input immutable. Output directories must be new; existing runs are never overwritten.
Use a source checkout with the `dev` dependencies because the safety gate runs pytest.
If the shell's `TMPDIR` points to a nonexistent directory, prefix `uv` with `TMPDIR=/tmp`.

## Layout and switches

The requested layout is retained: `src/data/{loader,features,api_simulator}.py`,
`src/models/grigoreva_gru.py`, `src/training/{offline,online}.py`, `src/metric.py`,
`src/cv.py`, and `experiments/ablations.py`. Supporting files provide config loading,
artifact persistence, and the CLI.

| Improvement | Independent config switch |
|---|---|
| Same-timestamp market averages | `features.market_average` |
| Rolling deviations and standard deviations | `features.rolling` |
| Auxiliary responder branches and losses | `model.auxiliary_targets` |
| Delayed daily gradient updates | `online.enabled` |
| Multiple random seeds | `ensemble.seed_ensembling` |

`baseline.yaml` disables all five. `grigoreva.yaml` enables features, auxiliaries, and
the six-member ensemble. `online.yaml` additionally enables online updates. Architecture
ensembling is separately controlled by `ensemble.architectures`; turning off seed
ensembling retains the first seed for **each** selected architecture. An ablation changes
one switch relative to the reference and resets every model, scaler, and stream state.
These switches test the author's reported improvements; they do not assume improvements
will appear on another dataset.

## Evaluation and causality

The score is `1 - sum(weight * (target - prediction)^2) / sum(weight * target^2)` over
`is_scored=True` rows. It uses float64 sums, without target centering, an added epsilon,
or clipping of the score. Predicting zero scores zero when the denominator is positive.
A zero denominator is explicitly undefined and raises. Fold aggregation pools numerator
and denominator across disjoint validation windows rather than averaging daily scores.

The simulator calls `predict(test, lags)` once per `(date_id, time_id)`. It waits for and
validates the returned Polars frame (`row_id`, `responder_6`, in the original row order)
before the next call. Test frames contain only the public API columns. All nine
responders for date `d-1` are delivered at **`(d, 0)`**, with columns named
`responder_N_lag_1` and `date_id=d`. Other timestamps receive `None`. A missing preceding
date produces an empty table; a day without `time_id=0` receives no lag release, matching
the distributed gateway implementation. No earlier responder within the current day is
available. Lag tables and evaluator truth are kept separate from the predictor.

Unscored rows still receive predictions, update recurrent/rolling state, and become
eligible for training after their labels are released. The model caches only the current
day's observed inputs and joins released labels by date, time, and symbol. Missing labels
get zero training weight while their input timesteps remain in the sequence. There is no
final-day update without a subsequent release. The first replay lag is visible, but the
new predictor has no matching cached replay inputs, so it does not repeat offline training
on the last training day.

Temporal splits use sorted unique dates, never shuffled rows. The Grigoreva configs
start at date 700 and use two 200-date validation windows. For dates 700–1698, these are:

| Fold | Offline training | Validation |
|---|---|---|
| 0 | 700–1298 | 1299–1498 |
| 1 | 700–1498 | 1499–1698 |

Set `cv.n_splits: 1` and `cv.gap_days: 200` for the Grigoreva private-period experiment:
fit through 1298, replay 1299–1498 as unscored warmup, and score 1499–1698. The gap is
**observed streaming time**, so updates on newly released gap-day labels are allowed.
It is not a 200-day embargo that discards the intervening market stream.

Offline fitting can use all labels in its declared historical partition. Auxiliary
forward shifts are supervision, never inputs. Their required endpoints must stay within
that partition; unavailable endpoint losses are masked. Fitted normalization means and
standard deviations never use validation. Each training run uses a fixed epoch budget
and does not automatically early-stop. Patrick's development studies use validation
scores to compare candidate budgets and online settings; those dates therefore serve
as development data. Five epochs is our current baseline, **not** a verified epoch
count from Patrick's submission. Use a separate chronological evaluation before
making generalization claims about selected settings.

## GRU reproduction

Both published architectures and the fixed 16-feature list are implemented. The full
pipeline has 125 inputs: 76 raw features (excluding 09–11), 16 deviations from rolling
means, 16 sample rolling standard deviations, 16 market means, and time. A rolling window
contains the last 1000 observed rows of that symbol, including the current row, and carries
across dates. Recurrent sequences are one day long and reset at the next day; hidden states
are keyed by symbol to support changing symbol sets.

The auxiliary version uses four **separate recurrent networks**, with outputs supervised
by responders 10, 9, 8, and 7, and a linear combiner predicting responder 6. Offline loss
adds five weighted normalized squared-error terms. Online learning uses only responder 6,
one AdamW step per released day, a fresh optimizer, learning rate 0.0003, weight decay 0.01,
and gradient clipping at 1.0. Predictions are clipped to [-5, 5] per member, then averaged.

See [source provenance and deliberate differences](docs/sources.md) for dimensions,
dropout rates, target formulas, source pins, and boundary corrections.

## Artifacts and limits

Each run records configuration, software versions, the safety-test output and code hash,
input file metadata, exact date splits, daily training losses, initial checkpoints,
prediction parquet files, pooled scores, and an online-update audit. Load a fresh callback
using `src.artifacts.load_predictor(path).predict`; it accepts the Kaggle-style Polars
signature. Checkpoints contain inference initialization state, including training feature
history, and are saved before validation adaptation. Those initial checkpoints do not
contain optimizer state. The Patrick plot runner separately saves resumable training
and day-boundary replay checkpoints, including optimizer state; see the runbook above.
For a deployment stream whose dates restart at zero, use
`load_predictor(path, reset_clock=True).predict` to reset the chronological cursor while
retaining rolling history. CV keeps the original historical dates and requires no rebasing.

Preparation and training read one day at a time and keep prepared arrays on disk. Full
competition runs still require substantial disk space, compute, and parquet scan I/O;
this is a correctness-first research implementation, not the author's optimized submission
runtime. CPU is tested. CUDA with deterministic algorithms passed the bounded L4
pilot; identical results across hardware/PyTorch versions are not promised.

The API batching logic was checked against the pinned distributed gateway on a synthetic
fixture. `scripts/verify_gateway.py` can repeat that differential check using a local copy
of the source. Normal tests use the resulting checked-in trace and need no network access.
The simulator does not reproduce Kaggle RPC, startup orchestration, or total notebook
runtime limits; its optional per-call timeout detects overruns after a call returns.
It is an information-flow boundary for this harness, not a sandbox against hostile Python
callbacks. Synthetic tests establish causal behavior on the tested cases, not universal
proof against arbitrary future code changes.
