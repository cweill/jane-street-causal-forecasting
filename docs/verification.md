# Verification record

Historical Patrick verification records follow. Current commands are in the README;
check counts below describe their recorded commits rather than the current suite.

## Patrick-only cleanup

Local verification after removing the other backend:

- Full suite: **115 passed**.
- Four retained deliberate leakage mutations: **all detected** by failing tests.
- Ruff lint/format and local Markdown links: passed.
- Default CLI synthetic smoke: both folds completed through the causal gate.
- A Patrick checkpoint created before the cleanup still loads; its replay predictions
  including online updates are exactly equal after the cleanup.
- Repeating its synthetic training after the cleanup produces identical model weights
  and frozen preprocessing state.

The model/training arithmetic and recorded experimental results were not changed.
Source fingerprints changed because preprocessing moved to its own module; old run
resumption uses the recorded source commit. Only irrelevant backend-specific tests
were removed; generic fitting-boundary and online-timing coverage was ported to Patrick.

## Official data download

Downloaded successfully on 2026-09-17 using
`kagglehub.competition_download('jane-street-real-time-market-data-forecasting')`.
The local symlink `data/competition` points to the Kagglehub cache. Pass
`--data data/competition/train.parquet` to the harness, rather than the parent
directory (which also contains example test and lag parquet files).

Metadata checks found 10 training partitions, 47,127,338 rows, 1,699 distinct
dates spanning 0–1698, 79 feature columns, and nine responder columns.
`artifacts/competition-data.json` records the paths, file sizes, schema counts,
date range, and official gateway SHA256.

The official gateway differs from the previously inspected mirror only in import
ordering. After inspecting that diff, its exact hash was added to the verification
allowlist. Running `scripts.verify_gateway` against the downloaded official file
matched all 15 synthetic batches. This is a batching/lag differential check; it does
not establish full RPC or hosted-runtime equivalence. At that initial download
checkpoint, training had not yet started; subsequent pilots are recorded below.

## Patrick local integration (2026-09-17)

All 61 tests pass, including training-only vocabularies and statistics, temporal
prefix and gradient causality, missing-asset masking, symbol permutation,
full-day versus cached inference, checkpoint replay, delayed online learning,
and independent seed averaging. The Patrick CLI smoke completed both temporal
folds after the mandatory safety gate. Both additional deliberate leakage
mutations were detected. GitHub Actions also validates the shared harness on Linux.

The [implementation decisions](patrick-implementation.md) distinguish recovered
slide settings from assumptions. No reported competition score is reproduced.

## Patrick CUDA pilot (2026-09-17)

The [Patrick L4 pilot](patrick-pilot-report.md) passed its 53-test remote gate,
full-size one-epoch training, matched frozen/online replays, and future-responder
perturbation checks. Measured work took 450.81 seconds with 4.29 GiB peak PyTorch
CUDA allocation. Independent score recomputation agreed within 7e-16. The
archive was verified and downloaded; Modal reported the App stopped with zero tasks.
This was before the later full-period experiments; that interval has since been inspected.

## Patrick cache and recovery checks (later on 2026-09-17)

The [runbook and machine-readable evidence](patrick-run-safety.md) record a
full-size L4 optimization benchmark and a separate CUDA interruption rehearsal.
Optimized preparation/replay matched original arrays, predictions and final weights
exactly. Restored training matched uninterrupted weights and loss history, including
dropout RNG state; restored online replay matched predictions and weights at lr=5e-4.
The recovery rehearsal's remote causal gate passed 76 tests. Subsequent local tests
cover non-finite losses/gradients without overwriting a healthy checkpoint. The full
local suite now passes 90 tests, and all five deliberate leakage mutations were detected.

`configs/patrick_ol.yaml` explicitly uses dates 1380–1698 for the requested plot,
with 1500–1698 as its primary scored interval. This reconstructs the author's
experiment under documented assumptions; it is not a fresh holdout for later tuning.

## Persistent submission repair (2026-09-18)

The first full plot attempt was cancelled before training; its CPU cache survived.
The [incident record](references/patrick-ol-interruption.json) includes the observed
cancellation times and local network error. The fixed launcher submits a durable
job to a deployed App and saves its FunctionCall ID instead of blocking locally.
The [integration probe](references/patrick-submission-rehearsal.json) remained
pending after its submitting process exited and subsequently completed. This tests
job lifetime separately from the earlier model/optimizer checkpoint rehearsals.

## W&B observer (2026-09-18)

The separate read-only CPU observer was connected to the running Patrick job.
W&B's API confirmed real optimization-loss history and training progress;
[connection evidence](references/patrick-wandb-monitor.json) records the check.
All 93 local tests and GitHub CI pass. Monitoring tests verify pooled evaluation
statistics, exclusion of warmup from scored cumulative R², checkpoint visibility,
true batch positions, and exclusion of model/optimizer contents from metric payloads.
The trainer image, lockfile, model configuration, and active job were not changed.
