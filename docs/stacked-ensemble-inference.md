# Patrick's stacked ensemble inference

The optional stacked implementation evaluates all ensemble members together at each
public timestamp. It stacks linear weights as `(models, output, input)`, uses
`einsum` for dense and GRU projections, batches asset attention over the model
axis, and caches each member's GRU state by layer and symbol. No loop over ensemble
members runs in its forward pass. This implements the two inference techniques
described in Patrick's supplied slides; it is our reconstruction, not his source code.

Add this to a Patrick configuration to enable it:

```yaml
inference:
  stacked_ensemble: true
ensemble:
  architectures: [patrick]
  seed_ensembling: true
  seeds: [0, 1, 2]
```

The default is `false`, retaining the per-model reference loop. The switch survives
model artifact serialization and daily replay checkpoints. For an already loaded
predictor, set `predictor.stacked_inference = True` before replay. Enabling this
switch does not create or train additional models.

All members must share architecture, categorical vocabularies, device, and dtype.
The stacked weights are an inference snapshot. Daily online updates still train
each original model with its own Adam state; the predictor invalidates the snapshot
after an update and rebuilds it before predicting. Code that changes weights outside
that update path must refresh the snapshot explicitly. This implementation adds a
stacked copy of the weights and therefore consumes additional GPU memory.

Only the current public timestamp reaches the inference engine. Missing symbols keep
their existing state, new symbols start at zero, and states reset at day boundaries.
The original previous-day label release and online update timing are unchanged.
Stacked outputs preserve original row order and average the members' responder 6
predictions. Online training remains sequential across members.

## Verification

`tests/test_stacked_ensemble.py` compares member predictions and hidden states with
the reference loop, covers both post-normalization settings, missing and new symbols,
online weight refresh, and daily checkpoint resume. It also perturbs future responders
and verifies that earlier predictions and updates remain unchanged. These tests are
part of the mandatory experiment safety gate.

Prediction equivalence uses `rtol=2e-5, atol=2e-6`: batched kernels can round
differently. GPU parity comparisons require matching full float32 precision:
`torch.set_float32_matmul_precision("highest")` **and**
`torch.backends.cudnn.allow_tf32 = False`. The matmul setting alone does not
disable TF32 inside the reference cuDNN GRU. The diagnostic found a prediction
difference around `1e-4` with cuDNN TF32 enabled, falling below `1e-6` when disabled.
The benchmark separately times the TF32 loop and reports its prediction difference;
it does not call those results strictly equivalent. These benchmark precision
settings do not change the existing running evaluation.

Under matching precision, online model weights and Adam state must remain bitwise identical to
the reference because inference does not alter the training calculations.

`scripts/modal_ensemble_benchmark.py` runs a bounded L4 benchmark on the existing
five-epoch checkpoint. It compares 1, 4, and 17 distinct perturbed model copies over
128 timestamps, then tests three members over two full days with real online updates
and a future-label perturbation. The copies measure correctness and throughput;
they are not independently trained seeds and do not measure ensemble forecasting gains.
The source volume is read-only and results use a separate benchmark directory.

The active single-seed plot reproduction remains on its frozen deployment. Adding
this implementation does not change its running workers or results.

## L4 measurements

The model-only benchmark used 128 timestamps with 36 symbols, alternating timing
order after warmup. Results are medians of two measurements per path. GRU weights
were flattened before timing to avoid penalizing the reference for copied storage.

| Members | Float32 loop | Stacked | Speedup | Speedup vs. TF32 loop |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.787 s | 1.762 s | 1.01× | 1.04× |
| 4 | 7.439 s | 1.747 s | 4.26× | 4.26× |
| 17 | 31.786 s | 1.788 s | 17.78× | 18.23× |

The maximum float32 prediction difference across these comparisons was `1.073e-6`;
the maximum hidden-state difference was `3.279e-6`. Both passed the combined absolute
and relative tolerances above. Building the 17-member stacked snapshot took 0.254 s.
These timings exclude preprocessing, daily online training, and checkpoint storage.
Single-member throughput is effectively unchanged. The observed scaling on this
small panel does not guarantee constant latency for arbitrary ensemble sizes.

The [precision diagnostic](references/patrick-stacked-precision.json) records the
original failed comparison and the controlled experiment identifying cuDNN TF32.
The [complete benchmark record](references/patrick-stacked-ensemble-benchmark.json)
includes the Modal call, implementation commit, matching local/remote source hash,
and the 89-test remote safety gate. The full local suite passed all 108 tests;
[CI passed](https://github.com/cweill/janestreet-repro/actions/runs/35403089331),
including leakage mutation checks.

The three-member online replay on dates 1380–1381 also passed: maximum prediction
difference `1.327e-6`, absolute pooled R² difference `6.409e-9`, and bitwise-identical
final weights and Adam state. Perturbing date 1381 responders changed neither
predictions through that date nor the final weights/optimizer. Its measured times
were 119.81 s for the loop and 40.90 s for stacked replay. These two-day timings are
secondary correctness-run observations: the copied reference GRUs emitted a storage
compaction warning, unlike the explicitly flattened model-only benchmark above.
They should not be used as a production latency guarantee.
