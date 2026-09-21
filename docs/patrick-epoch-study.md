# Offline epoch-budget study at fixed online LR

The existing development seeds improve in training R² through epoch five, while
their mean frozen validation R² is 0.01595035 at epoch three, 0.01609352 at epoch
four and 0.01552753 at epoch five. These are averages of individual-model metrics,
not ensemble scores. This motivates a controlled epoch comparison.

Train seeds **0, 1 and 2 through epoch four**, capturing immutable epoch-three and
epoch-four checkpoints. Reuse the completed epoch-five frozen and online-1e-4 results
from `ol-sweep-20260919T194601Z`. No other training parameter changes. Every online
comparison uses fresh Adam state, learning rate **1e-4**, persistent moments across
dates, betas (0.8, 0.95), three daily steps, clipping 1.0, and all nine targets.

Training is 0–1059, unscored online warmup 1060–1179, and scoring 1180–1379. Each
epoch's frozen and online predictions start from the same three-model checkpoint.
Different epochs intentionally have different weights; all six comparisons must
share scoring coverage, target energy, data provenance and chronological boundaries.
Scored labels become update targets only upon their next-day API release.

The original training, validation and replay caches mount read-only. CPU preflight
runs the causal tests, verifies the archived runtime, all cache checksums and the
reused epoch-five references. Training observations are compared against each
original seed's first four epochs, with absolute and relative tolerance 1e-7.
A discrepancy stops that worker; it is not silently accepted as an epoch effect.

The observer reconstructs exported models inside a CPU RNG isolation context.
Snapshots commit atomically and cannot be overwritten with different identities.
If interrupted after a training checkpoint but before exporting its epoch model,
resume recovers that artifact before training advances. Tests verify unchanged
training weights, optimizer state and RNG under observation and recovery, identical
first-four-epoch histories for four- and five-epoch budgets, and rejection of
mismatched paired weights.

The isolated Modal app `patrick-epoch-study` uses **three concurrent L4 training
workers**, followed by **four concurrent L4 replay workers** (epochs three and four,
each frozen and online). Checkpoints persist every 25 training days and each epoch;
replay checkpoints persist each day. No epoch-five retraining or replay is required.
The output volume is `janestreet-patrick-epoch-study`.

W&B records each seed's losses and frozen validation after every epoch. Its overview
compares all six rolling curves on explicit date axes, plus a pooled R² table by
epoch. The report includes ten scored 20-day blocks per pair. The key selection
metric is ensemble R² **after online adaptation**, not the training loss or the best
individual seed. This adaptive development study does not automatically select an
epoch, train 17 models, or begin another validation window.

Configuration: `configs/patrick_epoch_study.yaml`. Implementation:
`src/epoch_study.py` and `scripts/modal_epoch_study.py`. The earlier three-seed jobs
took roughly 48–52 minutes for five epochs; this four-epoch run also includes cache
verification and then roughly two hours of parallel replay. These are reference
durations, not guaranteed completion times.

Run `epoch-study-20260920T185609Z` launched from commit `32f722d` after all 128
local tests passed, all five deliberate leakage mutations were detected, and the
archived runtime comparison passed. The controller and W&B run have now completed successfully.

- [W&B overview](https://wandb.ai/cweill-self/janestreet-repro/runs/epoch-study-20260920T185609Z-overview)
- [Launch identity and durable call ID](references/patrick-epoch-study-launch.json)
- [Modal app](https://modal.com/apps/cweill/main/deployed/patrick-epoch-study)


## Completed results

All six evaluations cover the same 7,457,472 scored rows on dates 1180–1379,
with pooled target energy 10,547,832.643854462. Epoch-five results are reused from
the original development study; epochs three and four were newly trained and replayed.

| Offline epochs | Frozen R² | Online R² (LR 1e-4) | OL gain |
| --- | ---: | ---: | ---: |
| 3 | 0.01818019 | 0.02193983 | +0.00375964 |
| 4 | 0.01893456 | 0.02308571 | +0.00415114 |
| 5 | 0.01935094 | 0.02405157 | +0.00470063 |

Five epochs performs best among the tested budgets, both frozen and online.
Its online R² exceeds epoch four by 0.00096586 and epoch three by 0.00211174.
The earlier individual-model validation plateau did not translate into better
ensemble replay performance from stopping early. Keep five epochs as the working
baseline; this one adaptively reused development window does not establish a
universally optimal epoch budget.

- [Full result with checkpoint identities](references/patrick-epoch-study-result.json)
- [Rolling comparison](references/patrick-epoch-study-rolling.png)
- [Rolling values](references/patrick-epoch-study-rolling.csv)
- [Nonoverlapping scored blocks](references/patrick-epoch-study-blocks.csv)
