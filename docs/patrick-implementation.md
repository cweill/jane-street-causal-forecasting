# Patrick Yam reconstruction: implementation decisions

This implements the documented architecture inside the shared causal research
harness. It is a **reconstruction**, not a verified copy of Patrick's submission.
The [source tracker](patrick-yam-tracker.md) and
[slide transcriptions](patrick-yam-architecture.md) remain the provenance record.

## Implemented from the displayed settings

The model has 77 numerical inputs and three 16-dimensional categorical embeddings,
a 125→64 projection, eight residual blocks with eight-head asset attention,
one-layer temporal GRUs of width 256 projected back to 64, and a 64→256→128→9
prediction head. Feed-forward pre-gate width is 1024; dropout is zero. The
configuration uses RMSNorm, SiLU, and no final post-normalization.

Offline AdamW uses learning rate 0.0005, weight decay 0.0001, and betas
(0.95,0.9999). Online Adam uses betas (0.8,0.95) and three steps on the latest
released day's data. The multitask weights are [1,1,1,6,2,2,12,5,5].

Preprocessing fits only the declared offline dates. All-time global statistics,
category vocabularies and training targets never consume the validation partition.
Development validation can inform settings for later experiments. API replay, scored-row handling, lag release, zero-mean R²,
temporal folds, output validation, and artifact writing use the common research harness.

## Explicit reconstruction choices

| Unresolved source detail | Implemented choice |
|---|---|
| Numerical/category column mapping | Exclude feature_09/10/11 from the 76 raw numerical columns; use them as Cat1/2/3 |
| Category vocabulary and unknown values | Sorted unique training-only values; known capacities 83/13/540 plus one extra zero embedding for unknown/missing |
| Gaussian time endpoints | Normal quantile of `(time_id+0.5)/968`, epsilon 1e-4; out-of-range time IDs clip to boundary; no future-day length lookup |
| Numerical missing values and scale | Raw zero imputation before training-only sample-standard-deviation scaling; constant/all-missing columns get unit scale |
| Time feature normalization | Append Gaussian time after standardizing the 76 raw numerical columns |
| GLU/factory wiring | Split the FFN into value and gate halves, multiply value by SiLU(gate): 64→1024→512→64 |
| RMSNorm implementation | PyTorch RMSNorm with epsilon 1e-5 |
| Output head activations | SiLU between its linear layers |
| Post-normalization ablation | Optional RMSNorm after the complete encoder, before the head; exact author placement unverified |
| Target order | responder_0 through responder_8; main output index 6 |
| Detached loss balancing | Weighted SSE / weighted target energy per target, divided by its detached ratio with floor 1e-8; zero-energy targets skipped |
| Sample weighting | Multiply the normalized daily loss by `(200+d)/(200+max_train_date)` and optional ×1.5 for 968 distinct timestamps |
| Online learning rate | Historical plot configs use 0.0005; completed development and 17-model follow-up favor 0.0001 as the working baseline. Neither is source-verified. The earlier pilot used 0.0003. |
| Online optimizer lifetime | New Adam for online learning, persistent across days by default; daily reset independently configurable |
| Online target set | Same enabled responder targets/loss as offline; all nine released responders when auxiliary supervision is on |
| Epoch count, seed count | CPU starter: one epoch/one seed. Current full experiment: nine fixed epochs/17 seeds (`patrick_nine_epoch_ensemble.yaml`); the author's final epoch budget is unverified |
| Gradient clipping | Norm 1.0 for offline and online, configurable |
| Missing symbol observations | Mask asset attention, hold temporal state while absent, initialize new symbols at zero; empty timestamps produce finite masked outputs |
| Day boundaries | Reset recurrent state each day; discard prior-day cache after lag processing; no reuse of offline optimizer state |
| First replay update | Skip final offline day's labels when there are no matching cached replay inputs |
| Prediction clipping | None; source prediction clipping was not established |

Per-day sample multipliers are outside the normalized loss: applying the same
scalar to both its numerator and denominator would cancel the weighting. This is
an explicit effective-weighting reconstruction, pending the author's loss code.
Detached loss balancing makes the displayed optimization loss approximately
constant; it must not be used as a validation metric or early-stopping signal.

The predictor canonicalizes symbol order internally and restore API row order
on output, avoiding row-order-dependent floating-point differences. The model
also passes numerical permutation-equivariance tests without relying on sorting.

## Run and ablate

```bash
uv run --frozen js-repro run --data /path/to/development/train.parquet \
  --config configs/patrick.yaml --output artifacts/patrick-development
uv run --frozen js-repro ablations --data /path/to/development/train.parquet \
  --config configs/patrick.yaml --output artifacts/patrick-ablations
```

The config's 120 warmup / 200 scored dates are the documented count-based
scenario, not a claim that Patrick's executable split is known. Use a separately
bounded development dataset to avoid opening the final holdout during tuning.

Independent ablations cover auxiliary targets, online updates, seed ensembling,
recency weighting, full-length weighting, post-normalization, and GRU multiplier.
Loss balancing and daily optimizer reset can also be configured individually.

Patrick checkpoints use format version 2 and reload through
`src.artifacts.load_predictor` with `predict(test,lags)`. They initialize fresh replay;
they do not resume optimizer moments or a partially consumed day. Seed models remain
independent and their predictions are averaged. Optional
[stacked/einsum inference](stacked-ensemble-inference.md) batches seed weights and
caches recurrent states while retaining independent online optimizers.

## Required verification

The safety gate includes tests for future-feature prefix/gradient independence,
symbol permutation, empty/missing-asset masking, cached/full-day equivalence,
training-only statistics and vocabularies, unknown categories, checkpoint reload,
delayed-label timing, unreleased-responder perturbation, and seed independence.
Disposable mutation checks deliberately introduce future temporal mixing and
future-partition fitting and require those tests to fail.

The nine-day pilot uses a one-epoch/one-seed budget with paired frozen and
online replay checks. Its scores are execution
diagnostics, not a comparison to the author's 0.02059.
