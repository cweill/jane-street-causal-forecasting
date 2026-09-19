# Patrick validation during training

`python -m src.development` measures frozen-model weighted zero-mean R² after
**every epoch**. It logs `val/responder_6_r2` against `val/epoch`, alongside training
loss and training R². This is a new development workflow; the deployed 17-seed run
continues with its original fixed five-epoch protocol and final replay.

## Development split

| Dates | Use |
|---|---|
| 0–1059 | Fit preprocessing and train model parameters |
| 1060–1179 | Gap; no rows read by frozen epoch validation |
| 1180–1379 | Frozen validation after each epoch |
| 1380 onward | Excluded from fitting and validation |

These boundaries match the existing earlier development scenario. The new
`configs/patrick_epoch_validation.yaml` uses one seed, five epochs, and disables
online learning. `configs/patrick_development.yaml` retains its separate online
replay configuration. The command requires a fixed, single-seed Patrick split
ending before 1380. It never chooses an epoch or stops early automatically.

**Refit preprocessing for this split.** The original run's normalizer and prepared
arrays used dates through 1379; using them here would expose validation features
through fitted statistics and vocabularies. Training preparation uses a separate
cache identity. Validation is transformed once using the frozen development
normalizer and cached separately, with checksums and training-date provenance.
Source-file fingerprints may cover complete parquet files; rows outside the
declared partitions never enter preprocessing, training, or validation scoring.

This is a development set, not an untouched final test. Its dates were training
samples in the original experiment. Our already-inspected 1500–1698 replay also
cannot become a pristine holdout by renaming it. Any selected settings need
confirmation on another independently trained temporal split.

## Run

On a CUDA machine, with the competition parquet accessible:

```bash
TMPDIR=/tmp uv run --with wandb==0.30.0 python -m src.development \
  --data /path/to/train.parquet \
  --config configs/patrick_epoch_validation.yaml \
  --output artifacts/patrick-epoch-validation-001 \
  --cache artifacts/preparation-cache \
  --wandb-project janestreet-repro \
  --wandb-entity cweill-self \
  --wandb-id patrick-epoch-validation-001
```

W&B is optional: omit its arguments for local JSON observations only. Use
`WANDB_MODE=offline` to exercise logging without uploading. For CPU checks, use a
small configuration with `training.device: cpu`. The command runs the causal gate
before fitting; it does not launch or modify Modal apps.

Resume interrupted training with the identical command plus `--resume`. Keep the
same source files, config, code, cache, and W&B ID. Resume requires a committed
`training.pt`; an interruption during initial preparation needs a new output
folder. A changed validation cache or training identity is rejected. Observation
of a committed epoch is recovered before training continues.

## What is measured

Each day's forward pass resets recurrent state, uses only public numerical and
categorical inputs, and has online learning disabled. Whole-day evaluation uses
the model's causal forward path; tests compare it with timestamp-by-timestamp
API replay, including missing observations and scored-row masks. Frozen daily
state means the gap does not need to be replayed.

The metric pools float64 weighted SSE and target energy across **all scored rows**
in the validation period. It neither averages daily R² nor applies training
recency/full-length multipliers. An all-zero target-energy denominator yields an
undefined score (`null` in JSON; no R² point in W&B), with sufficient statistics
still recorded. Validation operates on a disposable model copy under inference
mode and RNG isolation, protecting training weights, gradients, modes, and state.

Artifacts:

- `training.pt`: recoverable model, optimizer, RNG, counters, validation identity,
  and epoch history; updated every 25 batches and at epoch boundaries.
- `history.json`: training and validation statistics for completed epochs.
- `epochs/epoch_N/`: immutable model and preprocessing checkpoint at each epoch,
  available for a later paired frozen/online replay with fresh online state.
- `identity.json`, `safety_gate.json`, cache manifests: split/config/source and
  preparation provenance.

Useful W&B charts:

- `val/responder_6_r2` versus `val/epoch`: held-out frozen-model performance.
- `train/epoch_responder_6_r2` versus `train/epoch`: pre-update training predictions.
- `train/epoch_mean_unbalanced_loss` versus `train/epoch`: meaningful training loss.

The training and validation R² series differ in both data and measurement time:
training pools predictions made while weights change, whereas validation uses the
fixed end-of-epoch weights. A widening gap is a diagnostic, not a direct numerical
estimate of overfitting. Balanced optimization loss remains unsuitable for epoch
selection. Online adaptation is evaluated separately after choosing candidate
epochs; this command does not imply frozen and online epoch rankings coincide.
