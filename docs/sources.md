# Source provenance and fidelity

Inspected on 2026-09-17. Algorithms were independently implemented; the author's source
was inspected in a separate temporary checkout and is not vendored into this project.

## Primary materials

- [Evgeniia Grigoreva, eighth-place write-up](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution), published July 14, 2025.
- [Author's repository](https://github.com/evgeniavolkova/kagglejanestreet), pinned commit
  `8598feb3ae28e470d1ed65661261f6f035a6e002`.
- [Competition data description](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data)
  and [evaluation](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview/evaluation).
  These pages did not expose text in the browsing tool. The actual distributed gateway
  and the competition data description rendered on a Kaggle notebook input page were
  used to check the callback interface.
- [Distributed `jane_street_gateway.py`, mirrored competition artifact](https://huggingface.co/datasets/TnnnT0326/Jane_Street_Competition/blob/4569a55/kaggle_evaluation/jane_street_gateway.py).
  This is a mirror of the competition's program, not an organizer-hosted repository.
  SHA-256: `d1aa2ab405f0d9b7e74a7c612e657ebcd632a4ab86e7fa8b80e517051e243984`.
- [Lag date-label clarification by the synthetic-data author](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/554604).

## Implementation mapping

The details below are checked against the pinned code, particularly
`janestreet/models/nn.py`, `janestreet/data_processor.py`, `janestreet/transformers.py`,
`scripts/python/run_cv.py`, and `scripts/python/run_test_gap.py`.

| Component | Implemented behavior |
|---|---|
| GRU 1 (`gru3`) | Separate GRU layers of widths 250, 150, 150; no dropout; linear scalar head |
| GRU 2 (`gru_mlp`) | GRU width 500, output dropout 0.3; Linear 500/ReLU/dropout 0.2; Linear 300/ReLU/dropout 0.1; scalar head |
| Auxiliary architecture | Four independent branches; linear 4-to-1 combination predicts responder 6 |
| Auxiliary branch order | responder 10, responder 9, responder 8, responder 7 |
| responder 9 | `r8[t] + r8[t+4]`, within a symbol's ordered observations |
| responder 10 | `r6[t] + r6[t+20] + r6[t+40]`, same observation ordering |
| Training | One day per batch; AdamW lr 0.0005, weight decay 0.01, gradient norm limit 1.0 |
| Online | Fresh AdamW lr 0.0003 each released day; one step; only primary-target loss |
| Seeds | 0, 1, 2 per architecture; arithmetic mean over six members |
| Market features | Unweighted mean of each selected feature across the current timestamp's available symbols |
| Rolling features | Current value minus trailing mean, and trailing sample standard deviation, window 1000 |
| Time feature | Raw time_id, clipped to fitted time range before standardization; bounds can expand from a released cached day during online updates |
| Normalization | Training observed-value means and sample standard deviations; missing input imputed with zero before scaling |
| Output | Each member clips responder-6 prediction to [-5, 5] before averaging |

The 16 selected features, in source order, are **06, 04, 07, 36, 60, 45, 56, 05, 51,
19, 66, 59, 54, 70, 71, 72**. The list is frozen from the write-up's code, not reselected
against local validation labels. This inherited research choice is not a nested feature
selection study; it should be disclosed when interpreting historical CV.

## Deliberate corrections and explicit choices

1. **Partition labels before deriving auxiliary targets.** The reference adds forward
   labels to the whole historical frame before CV splitting. That can consume validation
   responders at the boundary. Here, only training dates are loaded, and unavailable
   forward endpoints have zero auxiliary loss weight instead of becoming synthetic zero
   labels. Shifts can cross training-day boundaries if the endpoints remain within training.
2. **One causal rolling implementation.** The reference's training rolling path requires
   a full window; its fast submission path can average partial histories. Here both paths
   require 1000 valid observations for each feature's rolling statistics, use ddof=1,
   and ignore unavailable symbols rather than creating placeholder observations.
3. **Consistent missing-value policy.** Null/NaN/nonfinite feature values are treated as
   missing uniformly, and constant/all-missing columns get unit scale. Responders and
   weights in historical evaluation input must be finite. No responder imputation occurs.
4. **Ragged sequences.** Observed rows are sorted within each symbol/day; right padding
   receives zero weight. This replaces the source's rectangular reshape assumption.
   A missing timestep leaves that symbol's recurrent state untouched in inference.
5. **Fixed epochs.** Five epochs in the shipped configs is a declared research budget,
   not a recovered best checkpoint. Validation only measures the fitted predictor and
   its causal online adaptation. It never picks a checkpoint used to score earlier rows.
6. **Scoring degeneracies.** No epsilon is added to the official score. A zero denominator
   raises. A zero-energy training target contributes no normalized-loss term.
7. **CV indexing.** The prose lists fold 0 starting at 1298, but the code's 200-date split
   starts at 1299. This harness follows the code and records exact dates. Gap dates are
   replayed unscored, as in the author's gap experiment, rather than withheld entirely.
8. **API scope.** Historical replay retains original date IDs. The distributed local
   gateway requires its first date to be zero; the differential fixture satisfies that
   condition. Callback schema, ordering on chronological input, and lag release are
   verified. RPC and notebook wall-clock restrictions are outside the simulator.

## Repeat the gateway check

Obtain the pinned source from the mirror above, then run:

```bash
uv run --frozen python -m scripts.verify_gateway \
  --gateway-source /path/to/jane_street_gateway.py
```

The script refuses a different SHA-256. It executes the unmodified batching method
against temporary partitioned parquet, compares complete test/lag frame hashes for
every batch, and checks row-ID validation data. `tests/fixtures/gateway_trace.json`
contains the resulting 15-batch trace, including a missing time-zero case.
