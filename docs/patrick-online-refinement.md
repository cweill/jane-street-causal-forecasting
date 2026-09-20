# Lower-rate development refinement

This study adds four persistent-Adam online learning rates: **1e-5, 3e-5, 5e-5,
and 2e-4**. It reuses the three-seed initial checkpoint, cached replay days, frozen
baseline and completed persistent-Adam 1e-4 result from
`ol-sweep-20260919T194601Z`. There is no offline retraining and no baseline rerun.

All settings other than learning rate stay fixed: seeds 0–2, five offline epochs,
training dates 0–1059, replay 1060–1379, scored dates 1180–1379, three online steps,
betas (0.8, 0.95), clipping 1.0, and all nine targets. Each new replay starts with a
fresh online optimizer and updates only after previous-day labels are released.

The isolated `patrick-online-refinement` Modal app mounts the original study volume
read-only. Its CPU coordinator runs the causal gate, verifies cache checksums,
checks that the model/data/update implementation and dependency lock match the
original source archive, and validates both reused reference results. The new
report records each trial's actual provenance; reused identities are not rewritten.
The summary API accepts an explicit expected provenance per trial while retaining
the stricter same-provenance default used by existing studies.

Four independent L4 jobs perform the new replays, with day-boundary checkpoints,
durable call IDs and retries. Each first day's predictions must exactly match the
reused frozen baseline before adaptation can continue. The separate 17-model
follow-up remains unchanged; if still running, total concurrency can reach five GPUs.

W&B reports all six curves on explicit date axes and pooled weighted zero-mean R².
The final report also includes **ten non-overlapping 20-day scored blocks**, each
with R² and its difference from frozen. Blocks use evaluator-owned primary scoring
statistics and require matching row counts and target energy for every date.
Block diagnostics do not replace the pooled metric or provide independent samples
for a formal significance claim.

This is an adaptive development search prompted by the first sweep. It does not
create an untouched test set, change defaults, or automatically launch a winner.
The earlier development online replays took roughly 92–115 minutes each; the four
new jobs run concurrently, subject to GPU availability and verification overhead.

Implementation: `src/online_refinement.py`, `src/online_sweep.py` and
`scripts/modal_online_refinement.py`. Output volume:
`janestreet-patrick-online-refinement`.

Run `ol-refine-20260920T075437Z` launched from commit `2cb18ef`. Before launch,
all 126 local tests passed, all five leakage mutations were detected, and the
original archived runtime matched all 27 checked files and five replay definitions.
The remote causal gate passed 107 tests and CI passed. All four GPU workers started,
passed exact first-day parity against the reused frozen run on date 1060, and had
completed 1–3 replay days when verified. Results remain pending.

- [W&B overview](https://wandb.ai/cweill-self/janestreet-repro/runs/ol-refine-20260920T075437Z-overview)
- [Launch identity and durable controller call](references/patrick-online-refinement-launch.json)
- [Modal app](https://modal.com/apps/cweill/main/deployed/patrick-online-refinement)
