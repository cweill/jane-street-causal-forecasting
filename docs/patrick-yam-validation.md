# Patrick Yam: validation boundary audit

Checked 2026-09-17. Exact executable split: **not yet established**.

## Primary evidence

[Presentation PDF, page 29](https://drive.google.com/file/d/1AdzXrOZN699SGHAyb32slLBkEMxBCYiJ/view)
prints these ranges:

| Role | Printed date_id range | Intended use |
|---|---|---|
| Offline training | 0–1379 | Fit initial model |
| Simulated public period | 1380–1499, described as 120 dates | Online learning |
| Simulated private period | 1500–1699, described as 200 dates | Online learning and evaluation |

The [archived PDF rendering](references/patrick-yam/evaluation-split.png) preserves
the exact slide. In the [video at 12:29–13:17](https://www.youtube.com/watch?v=lfzzPZZyzjE&t=749s),
the author describes the same order: offline fit, 120 intervening days used for
online adaptation, then about 200 days used for adaptation and scoring. He does
not speak the numeric date endpoints. Automatic-caption timestamps are retained
in `artifacts/references/patrick-yam/transcript.md`.

The online-ablation plot starts at 1380, consistent with replay beginning there.
It is a rolling diagnostic that also spans the warmup; it does not establish that
warmup predictions contribute to the reported aggregate validation score.

PDF page 30 separately discusses a last-120-date validation holdout for model
selection. PDF page 32 says the fixed epoch count is chosen using performance
after online learning and the final model is retrained on all available data.
These statements do not provide executable boundary predicates or identify the
precise run behind the reported 0.02059 score.

## The one-day discrepancy

Downloaded competition metadata establishes **1,699 distinct dates, 0–1698**.
The slide's inclusive ranges instead span 1,700 dates. The data contains no date
1699. Two reasonable local interpretations are:

| Scenario | Offline training | Unscored warmup | Scored dates | Counts: train / warmup / scored |
|---|---|---|---|---|
| Preserve printed starting boundaries; intersect with available data | 0–1379 | 1380–1499 | 1500–1698 | 1380 / 120 / 199 |
| Preserve 120 warmup and final 200 scored dates | 0–1378 | 1379–1498 | 1499–1698 | 1379 / 120 / 200 |

Both rows are **our explicit interpretations**, not confirmed author code. The
second is the count-based split previously proposed in conversation. It must not
be described as Patrick's exact split. Machine-readable boundaries and counts are
saved in [validation-scenarios.json](references/patrick-yam/validation-scenarios.json).
These are documentation scenarios, not active training configurations.

Do not try both on the final holdout and retain whichever approaches 0.02059.
Choose a protocol before running models. If a boundary-sensitivity check is later
authorized, report both complete results without selecting a winner by closeness
to the published score.

## Scoring and label-release contract

For a local comparison, use the same scenario for both methods. Accumulate weighted
zero-mean R² on responder_6 over scored rows only, pooling the numerator and
denominator across the scored period. Warmup affects model state but contributes
nothing to that aggregate metric. This is our competition-metric contract; the
author's exact local score implementation remains unavailable.

At time zero of the first scored date, the preceding warmup day's labels are
available for an update before the first scored prediction. Later scored-day
labels may affect only predictions after their next-day release. In particular,
future scored dates cannot be used for normalization, checkpoint selection, or
initial offline training.

The first warmup update is another unresolved reproduction detail: the API can
release the final offline day's labels, but our current predictor has no cached
inputs for that day and skips that update. Whether Patrick caches those inputs
and retrains on that day needs source verification.

## What the reported scores can establish

The results slide reports local 0.02059 and public leaderboard 0.01303 for its final
listed experiment. These are reference values from the presentation, not verified
final-submission scores. Matching them requires more than date boundaries: epoch
count, seed ensemble, online learning rate and optimizer lifetime, preprocessing,
loss implementation, and score aggregation still matter.

Targeted public searches for the author's split and score did not locate executable
author code resolving the discrepancy. Unrelated community reproductions are not
evidence of his original split. Current conclusion: the temporal protocol and
printed dates are documented; exact comparability to 0.02059 remains unverified.
