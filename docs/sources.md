# Source provenance and fidelity

Patrick's architecture is independently reconstructed from his public presentation
and the user-supplied transcript and slide snapshots. His executable submission
code is not vendored. The [source tracker](patrick-yam-tracker.md),
[architecture transcription](patrick-yam-architecture.md), and
[implementation decisions](patrick-implementation.md) record what is confirmed,
what is inferred, and what remains unresolved. Slide excerpts retain their attribution.

## Competition protocol sources

- [Competition data description](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data)
  and [evaluation](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview/evaluation).
  These pages did not expose text in the browsing tool. The actual distributed gateway
  and the competition data description rendered on a Kaggle notebook input page were
  used to check the callback interface.
- [Distributed `jane_street_gateway.py`, mirrored competition artifact](https://huggingface.co/datasets/TnnnT0326/Jane_Street_Competition/blob/4569a55/kaggle_evaluation/jane_street_gateway.py).
  This is a mirror of the competition's program, not an organizer-hosted repository.
  SHA-256: `d1aa2ab405f0d9b7e74a7c612e657ebcd632a4ab86e7fa8b80e517051e243984`.
- [Lag date-label clarification by the synthetic-data author](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/554604).

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
