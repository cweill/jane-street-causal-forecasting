# Jane Street Causal Forecasting

A leakage-safe research harness for reconstructing **Patrick Yam's second-place
Jane Street Real-Time Market Data Forecasting method**. It implements training-only
preprocessing, asset attention with temporal GRUs, delayed online updates, seed
ensembles, and local evaluation through the competition's information flow.

**Correctness comes before experiments:** every CLI training/ablation run first
executes metric, protocol, and causal tests. This is an independent reconstruction,
not the author's submission code or a verified reproduction of his leaderboard score.
The [implementation decisions](docs/patrick-implementation.md) and
[source tracker](docs/patrick-yam-tracker.md) distinguish recovered settings from assumptions.

## Current Patrick results

The current best tested configuration is **nine offline epochs, 17 seeds, and
online LR `1e-4`**. It uses three daily Adam steps, betas `(0.8, 0.95)`, persistent
online optimizer state, and all nine responder targets. Offline training LR is
`5e-4`. Online updates use only previous-day labels after their API release.
The [completed nine-epoch run](docs/patrick-nine-epoch-ensemble.md) achieved
**0.02077608 online R²**, up from **0.01941540** at five epochs.

Completed experiments report pooled weighted zero-mean R²:

| Experiment | Seeds | Scored dates | Frozen R² | Online R² at `1e-4` |
|---|---:|---|---:|---:|
| Development, 3 offline epochs | 3 | 1180–1379 | 0.01818019 | 0.02193983 |
| Development, 4 offline epochs | 3 | 1180–1379 | 0.01893456 | 0.02308571 |
| Development, 5 offline epochs | 3 | 1180–1379 | 0.01935094 | 0.02405157 |
| Development, 7 offline epochs | 3 | 1180–1379 | 0.02040328 | 0.02546736 |
| Development, 9 offline epochs | 3 | 1180–1379 | 0.01962301 | 0.02568815 |
| Earlier confirmation, 7 offline epochs | 3 | 980–1179 | 0.01364799 | 0.02634289 |
| Earlier confirmation, 9 offline epochs | 3 | 980–1179 | 0.01522683 | 0.02725828 |
| Later follow-up, 5 offline epochs | 17 | 1500–1698 | 0.01412888 | 0.01941540 |
| Later follow-up, 9 offline epochs | 17 | 1500–1698 | **0.01432650** | **0.02077608** |

Development models train on dates 0–1059 and replay unscored warmup 1060–1179.
Earlier-confirmation models train on 0–859 and warm up on 860–979.
Both 17-model ensembles train on 0–1379 and replay unscored warmup 1380–1499.
Scores from different windows are not directly comparable.

![Nine-epoch 17-model ensemble: online learning at LR 1e-4 versus frozen, rolling 20-day weighted zero-mean R²](docs/references/patrick-nine-epoch-best-vs-frozen.png)

Both curves start from the same nine-epoch ensemble. They show rolling 20-day
weighted zero-mean R²; the dotted line marks scoring starting at date 1500.
Over dates 1500–1698, online learning adds **0.00644958** R² over frozen.
Compared with the five-epoch online ensemble, nine epochs adds **0.00136068**
and improves all ten nonoverlapping scored blocks (nine 20-day blocks and one
19-day block). Paired verification passed on all **7,397,456 scored rows**.
The [runbook](docs/patrick-nine-epoch-ensemble.md) includes the four-curve comparison,
[exact results](docs/references/patrick-nine-epoch-ensemble-results.json), and
[block scores](docs/references/patrick-nine-epoch-ensemble-scored-blocks.csv).

The initial [3/4/5-epoch study](docs/patrick-epoch-study.md) favored five epochs;
the [longer-training study](docs/patrick-long-epochs.md) and
[earlier-split confirmation](docs/patrick-epoch-confirmation.md) then favored nine.
The [learning-rate refinement](docs/patrick-online-refinement.md) found `5e-5`
narrowly ahead of `1e-4` on the five-epoch development window (0.02406497 versus
0.02405157), so the learning-rate optimum is not established. We retained `1e-4`
for the nine-epoch comparison to isolate the training-budget change.

Our later-window score is numerically above Patrick's reported 0.02059, but the
exact reproduction details remain unconfirmed. These windows have been inspected
during development and are **not untouched test sets**. This result does not
establish leaderboard equivalence or a statistically significant advantage.

Use `configs/patrick_nine_epoch_ensemble.yaml` and its
[runbook](docs/patrick-nine-epoch-ensemble.md) for the current full experiment.
Historical configs retain their original settings: `configs/patrick_ensemble.yaml`
uses five epochs and online LR `5e-4`, and its follow-up launcher overrides that
rate to `1e-4`. `configs/patrick.yaml` remains a small CPU starter. Recorded launch
identities pin the source and settings for reproducing each experiment.

[Latest W&B results and comparison chart](https://wandb.ai/cweill-self/janestreet-repro/runs/nine-epoch-ensemble-20260923T200241Z-overview) ·
[Historical five-epoch OL comparison](https://wandb.ai/cweill-self/janestreet-repro/runs/ol-followup-20260920T070707Z).

## Workflows and implementation

The [bounded Modal GPU pilot](docs/modal-pilot.md) runs the real-data correctness
checks on dates 700–708 and downloads its checkpoints, audits, and runtime report.
The [Patrick L4 pilot passed](docs/patrick-pilot-report.md), including a real-data
future-responder perturbation check.
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
The completed [longer-training study](docs/patrick-long-epochs.md) found the best
online score at nine epochs, with only a small lead over seven. The
[earlier-split confirmation](docs/patrick-epoch-confirmation.md) compares seven
and nine using fresh models and preprocessing fitted on dates 0–859.
The completed [nine-epoch 17-model run](docs/patrick-nine-epoch-ensemble.md) followed
these development comparisons, passing the three-seed pilot, all epoch-five
weight controls, and the final matched frozen/online evaluation.

## Run locally

Python 3.12 is the tested runtime. From this checkout:

```bash
uv sync --python 3.12 --frozen --extra dev
uv run --frozen js-repro verify
uv run --frozen python -m pytest -q
uv run --frozen python -m scripts.check_leakage_mutations

# Small synthetic runs; reduced network dimensions, one epoch, no financial meaning.
uv run --frozen js-repro smoke --config configs/patrick.yaml --output artifacts/patrick-smoke

# Real research: supply the competition training parquet downloaded from Kaggle.
uv run --frozen js-repro run --data /path/to/train.parquet \
  --config configs/patrick_development.yaml --output artifacts/patrick-cv

# Reference plus seven independent ablations, using identical folds and epoch budgets.
uv run --frozen js-repro ablations --data /path/to/train.parquet \
  --config configs/patrick_development.yaml --output artifacts/ablations
```

`--data` accepts a parquet file or a directory containing parquet partitions. Keep the
input immutable. Output directories must be new; existing runs are never overwritten.
Use a source checkout with the `dev` dependencies because the safety gate runs pytest.
If the shell's `TMPDIR` points to a nonexistent directory, prefix `uv` with `TMPDIR=/tmp`.

## Layout and switches

The core modules are `src/data/{loader,api_simulator,patrick_features,normalization}.py`,
`src/models/{patrick_yam,patrick_ensemble}.py`, `src/training/patrick.py`, `src/metric.py`,
and `src/cv.py`. Configs and scripts cover local research and persistent Modal runs.

| Ablation | Config setting |
|---|---|
| Nine responder targets versus primary target only | `model.auxiliary_targets` |
| Delayed daily gradient updates | `online.enabled` |
| Multiple random seeds | `ensemble.seed_ensembling` |
| Recency weighting | `training.recency_weighting` |
| Full-length-day weighting | `training.full_length_weighting` |
| Final normalization | `model.post_norm` |
| Temporal GRU capacity | `model.rnn_multiplier` |

`experiments/ablations.py` changes one setting at a time relative to the reference,
resetting models, preprocessing, and stream state. Optimizer reset policy and loss
balancing are additionally configurable. These interventions test hypotheses;
improvement on another interval is not assumed. `configs/patrick.yaml` is a one-epoch
CPU starter; the fixed development and ensemble configs describe larger GPU experiments.

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

Unscored rows still receive predictions, update recurrent state, and become
eligible for training after their labels are released. The model caches only the current
day's observed inputs and joins released labels by date, time, and symbol. Missing labels
get zero training weight while their input timesteps remain in the sequence. There is no
final-day update without a subsequent release. The first replay lag is visible, but the
new predictor has no matching cached replay inputs, so it does not repeat offline training
on the last training day.

Temporal splits use sorted unique dates, never shuffled rows. The fixed Patrick
protocols are:

| Protocol | Offline training | Unscored warmup | Scored replay |
|---|---|---|---|
| Development | 0–1059 | 1060–1179 | 1180–1379 |
| Later plot reproduction | 0–1379 | 1380–1499 | 1500–1698 |

Warmup is observed streaming time: updates on newly released labels are allowed.
The 120-day warmup is an experimental choice, not an API constraint. The actual
training data ends at 1698; the later scored interval contains 199 dates.

All nine responder targets are supervision, never model inputs. Fitted normalization
and category vocabularies use only offline training dates. Every run uses a fixed
epoch budget without automatic early stopping. Development validation informs later
hyperparameter choices; it is not an untouched holdout. Nine epochs is our current
best tested budget, not a verified epoch count from Patrick's submission.

## Model and features

Inputs are 76 raw numerical features (excluding 09–11), a Gaussian time-of-day
feature, and embeddings for categorical features 09–11. Numerical statistics and
category vocabularies are fitted on training dates and frozen. The model combines
same-timestamp asset attention with causal temporal GRUs and a nine-target head.
Recurrent state resets each day and is keyed by symbol to handle missing observations.

The full configuration uses eight blocks, model width 64, eight attention heads,
and GRU multiplier four. Online learning starts a separate Adam optimizer and updates
from the latest released day; seed predictions are averaged. Stacked inference batches
weights across seeds and caches GRU states, while online optimizers remain independent.

See [source provenance](docs/sources.md) and
[implementation choices](docs/patrick-implementation.md) for exact recovered settings
and the remaining reconstruction assumptions.

## Artifacts and limits

Each run records configuration, software versions, the safety-test output and code hash,
input file metadata, exact date splits, daily training losses, initial checkpoints,
prediction parquet files, pooled scores, and an online-update audit. Load a fresh callback
using `src.artifacts.load_predictor(path).predict`; it accepts the Kaggle-style Polars
signature. Checkpoints contain inference initialization state and frozen preprocessing,
and are saved before validation adaptation. Those initial checkpoints do not
contain optimizer state. The Patrick plot runner separately saves resumable training
and day-boundary replay checkpoints, including optimizer state; see the runbook above.
A freshly loaded Patrick predictor also accepts a deployment stream whose dates
restart at zero; it has no prior replay clock. CV retains historical date IDs.

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


Historical experiments pin their source commits and cache/checkpoint identities.
The Patrick-only cleanup changes source fingerprints: use the recorded commit for
resuming or exactly reproducing an old job. New runs build fresh cache identities;
identity checks are not bypassed. Patrick format-2 inference checkpoints remain loadable.

See the [publication audit](docs/publication-audit.md) for scan scope, credential
handling, and identifying metadata retained in Git history. Competition data, trained
weights, credentials, and local run directories are not included in this repository.
