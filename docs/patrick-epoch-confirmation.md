# Earlier-split confirmation: seven versus nine epochs

This experiment checks whether nine epochs' small online advantage over seven
persists on a second chronological split. The preceding development study found
online R² of 0.02546736 at seven epochs and 0.02568815 at nine; nine won six of ten
nonoverlapping 20-day blocks. Seven beat five in all ten blocks.

| Partition | Dates |
|---|---|
| Offline training and preprocessing fit | 0–859 |
| Unscored online warmup | 860–979 |
| Scored replay | 980–1179 |

Train seeds 0, 1 and 2 from scratch through nine epochs, exporting immutable
checkpoints at seven and nine. Every other setting matches
`configs/patrick_long_epochs.yaml`: offline LR 5e-4, the same architecture,
auxiliary targets, sample weights, and optimizer parameters. Online learning uses
LR 1e-4, three daily steps, and a fresh Adam optimizer with betas (0.8, 0.95),
whose state persists across replay days.

Preprocessing is fitted only on 0–859. Training, validation, and daily replay
caches are built for this split from the checksum-verified raw dataset. Earlier
checkpoints or normalizers are not reused. Previous-day responder releases follow
the simulator; the first replay day has no cached previous-day feature inputs for
an online update, matching the preceding study's policy.

The same-seed frozen and online ensembles are evaluated over exactly the same
320 replay days. The primary measure is pooled weighted zero-mean responder-6 R²
on the 200 scored days; the report also includes rolling curves and ten
nonoverlapping 20-day blocks. Pairing checks require identical initial weights,
scored rows, target energy, and first-day predictions, and unchanged frozen weights.

This is temporal confirmation, not a pristine final holdout: these historical
dates appeared in earlier experiments' training or warmup intervals. No prior
models cross this run's training boundary. Both candidates are reported without
automatic winner promotion or 17-seed retraining. Prefer nine only if its advantage
is consistent across splits and blocks; mixed evidence favors the cheaper seven.

## Execution and recovery

Configuration: `configs/patrick_epoch_confirmation.yaml`.
Launcher: `scripts/modal_epoch_confirmation.py`.
App: `patrick-epoch-confirmation`.
Output volume: `janestreet-patrick-epoch-confirmation`.

A durable CPU controller verifies source/config identities, runs the causal gate,
and prepares and verifies the caches before requesting GPUs. Three L4 training
workers run concurrently, followed by four L4 replay workers. Training checkpoints
are committed every 25 days and at epoch boundaries; replay commits daily.
Epoch exports preserve training RNG state and recover after interruption.
W&B records train losses, epoch validation, date-aligned replay metrics, and final
comparison charts. Allow roughly 3–5 hours including preparation and scheduling.

## Launch

Started September 23, 2026 UTC (September 22 Pacific), with source commit
`c657eecd071ec62bef85e7357c1e1505c68f7a1e`.

[W&B overview](https://wandb.ai/cweill-self/janestreet-repro/runs/epoch-confirmation-20260923T065042Z-overview) ·
[Launch record](references/patrick-epoch-confirmation-launch.json).

Before launch, 117 local tests passed and all four deliberate leakage mutations
were detected. The raw dataset remains in the read-only research volume.

## Completed replay results

| Epochs | Frozen R² | Online R² |
|---|---:|---:|
| 7 | 0.01364799 | 0.02634289 |
| 9 | 0.01522683 | 0.02725828 |

All four replay workers completed. Nine epochs improves online R² by
0.0009153861 and wins all ten nonoverlapping scored 20-day blocks. Each candidate
uses 7,174,816 scored rows and target energy 9,482,158.852753779. These completed
replay artifacts motivate the [staged full ensemble run](patrick-nine-epoch-ensemble.md).
