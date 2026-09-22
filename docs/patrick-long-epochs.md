# Longer offline training: epochs 5, 7 and 9

This development experiment holds seeds 0–2, architecture, optimizer parameters,
preprocessing, and chronological boundaries fixed. It trains each seed through
nine epochs, saves immutable artifacts at epochs 5, 7 and 9, and evaluates each
three-model ensemble both frozen and with online LR **1e-4**. Online Adam starts
fresh, persists across days, and uses three daily steps with betas (0.8, 0.95).
No 17-seed retraining or automatic winner promotion is included.

| Partition | Dates |
|---|---|
| Offline training | 0–1059 |
| Unscored online warmup | 1060–1179 |
| Scored replay | 1180–1379 |

The primary comparison is pooled weighted zero-mean responder-6 R² after online
adaptation. The report includes matched frozen scores, six rolling 20-day curves,
and ten nonoverlapping scored blocks per pair. This is an adaptively reused
development window; another independently trained split remains a later check.

## Controls and recovery

The Patrick-only cleanup changed preprocessing fingerprints. This study therefore
rebuilds training and validation caches under the current source identity, from
the checksum-verified immutable raw dataset. It does not rewrite old checkpoint
signatures or resume old optimizers across source changes. The existing daily
replay cache is read-only and every day is checksum-verified before GPU launch.

Rebuilt normalization/vocabularies must match the earlier development run. Each
seed's first five epoch histories must match that run within the established
1e-7 tolerance, and its epoch-five weights must match exactly. Fresh epoch-five
frozen and online replay scores must agree with the previous controls within
1e-10, with identical initial weights, scored row counts, and target energy.
Any failed control stops the study instead of reporting an epoch improvement.

Training commits resumable state every 25 days and at epoch boundaries. Epoch
exports preserve RNG state and are immutable; interruption between training-state
commit and artifact export is recovered on resume. Replay commits at each day
boundary. The CPU coordinator submits durable remote calls and logs explicit
training-epoch and replay-date axes to W&B.

App: `patrick-long-epochs`. Output volume: `janestreet-patrick-long-epochs`.
The maximum concurrency is three L4 training workers followed by four L4 replay
workers; the six replay jobs run in two waves. Allow roughly 4–6 hours including
fresh preparation, training, and replays, subject to cache I/O and GPU availability.
No GPU is requested until the CPU causal gate and preparation checks pass.

Configuration: `configs/patrick_long_epochs.yaml`.
Launcher: `scripts/modal_long_epochs.py`.
The original control run is `ol-sweep-20260919T194601Z`.

## Launch

Started September 22, 2026, with source commit `63757f150fed6dcd61dd7d938b575096a0a1b631`.

[W&B overview](https://wandb.ai/cweill-self/janestreet-repro/runs/long-epochs-20260922T184410Z-overview) ·
[Immutable launch record](references/patrick-long-epochs-launch.json).

Local verification before launch: 116 tests passed and all four deliberate
leakage mutations were detected. Results will be recorded after all controls
and paired replays finish.
