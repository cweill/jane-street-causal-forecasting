# Patrick Yam reproduction tracker

Updated: 2026-09-17. Status: reconstruction implemented; local causal and integration tests pass.
The [bounded L4 pilot](patrick-pilot-report.md) also passed; predictive comparison
and recovery of unresolved source details remain outstanding.

This is the entry point for the supplied transcript and ten screenshots. Values below
are author-reported or transcribed unless marked as derived. None of the reported
scores or runtime measurements are local reproduction results.

- [Implemented reconstruction and explicit assumptions](patrick-implementation.md)
- [Architecture diagram, block code, and full displayed configuration](patrick-yam-architecture.md)
- [Source notes and evaluation constraints](patrick-yam-source-notes.md)
- [Validation boundary audit and explicit local scenarios](patrick-yam-validation.md)
- [Video](https://www.youtube.com/watch?v=lfzzPZZyzjE)
- [Author's slides](https://drive.google.com/file/d/1AdzXrOZN699SGHAyb32slLBkEMxBCYiJ/view)
- [Screenshot manifest with SHA256 hashes](references/patrick-yam/manifest.json)
- [Reported results as CSV](references/patrick-yam/reported-results.csv)

The original screenshots are copied into `docs/references/patrick-yam/`, outside the
ignored artifacts directory. They do not depend on the temporary CleanShot locations.
The downloaded PDF and automatic captions remain in
`artifacts/references/patrick-yam/`; that directory is ignored by git.

## Evidence index

| Screenshot | Recorded detail | Qualification |
|---|---|---|
| [Architecture](references/patrick-yam/architecture.png) | 77 numerical inputs, clipping [-10,10]; categories 83/13/540, embedding width 16; model width 64; eight blocks; head 64→256→128→9 | Input projection width 125 is derived; category-column mappings unknown |
| [Block code](references/patrick-yam/transformer-block.png) | Asset attention, one-layer temporal GRU, projection, gated FFN; three pre-normalized residual branches | Constructor defaults are not the displayed experiment configuration |
| [Sample weighting](references/patrick-yam/sample-weighting.png) | All training samples; multiplier `(200+d)/(200+d_max)`; another ×1.5 for length 968 | Combination with row weights and definition of sequence length need code |
| [Configuration](references/patrick-yam/configuration.png) | Eight heads; FFN width 1024; GRU multiplier 4; dropout 0; RMSNorm; SiLU; AdamW lr 5e-4, wd 1e-4, betas (0.95,0.9999) | GRU width 256 is derived; factory/gating wiring unknown |
| [Fixed epochs](references/patrick-yam/fixed-epochs.png) | One shared epoch count across seeds; separate best-epoch selection was noisy; mentions last 120 dates as validation | Epoch count absent; this 120-date holdout differs from the warmup experiment |
| [Seed ensemble](references/patrick-yam/seed-ensembling.png) | More seeds improved reported score, with diminishing gains | Exact graph coordinates not transcribed |
| [Online settings](references/patrick-yam/online-settings.png) | Three Adam steps per newly labeled day; betas (0.8,0.95); latest labeled day only | Online learning rate and optimizer persistence across days unknown |
| [Online ablation](references/patrick-yam/online-ablation.png) | Blue with updates versus orange without; rolling 20-day score; frozen model deteriorates | Plot says “time id” starting 1380; interpretation as date_id is inferred |
| [Fast inference](references/patrick-yam/fast-inference.png) | Stack seed weights, einsum linear layers, cache GRU states; 17 models plus online learning in 2.5 hours | Author-reported runtime; hardware and exact cache implementation unverified |
| [Reported results](references/patrick-yam/reported-results.png) | Local and public leaderboard scores improve in the same order | Appears cumulative; not independent ablation evidence |

## Training and inference details

The full configuration is transcribed in the architecture document. Additional
settings include no scheduler, gradient accumulation 1, `use_post_norm=False`, and
target weights `[1,1,1,6,2,2,12,5,5]`. Verify output ordering before assigning each
weight to a responder. The transcript describes all nine responder targets and
per-target loss scaling by its detached value; the exact loss and safeguards remain open.

Numerical preprocessing uses training-fitted global mean and standard deviation.
The additional time-of-day feature is uniformly rescaled, then Gaussian-transformed.
The configuration's `standardize_input=False` may refer to standardization inside
the model and does not establish that external preprocessing is absent.

Online learning starts without the offline optimizer state. **This does not establish
a daily optimizer reset.** At `(d, 0)`, our causal interpretation is to join newly
released responders from `d-1` with cached inputs from `d-1`, perform the updates,
then predict the current timestamp. The author reports that reusing the offline
optimizer and replaying multiple older days hurt his results.

The stacked linear layer can be expressed as:

```python
# N independent models; I and J are the remaining input axes.
# x: [N, I, J, Din], weight: [N, Dout, Din], bias: [N, Dout]
y = torch.einsum("nijx,nyx->nijy", x, weight)
y = y + bias[:, None, None, :]
```

Each seed keeps separate weights and hidden states. Final predictions are averaged;
weights themselves are not averaged. The complete stack, attention, normalization,
GRU, state resets, and updates must preserve equivalence with separate-model inference.

## Author-reported score progression

| Change | Local validation | Public leaderboard |
|---|---:|---:|
| First model | 0.01320 | 0.00872 |
| Tuned hyperparameters | 0.01695 | 0.01103 |
| Sample weighting scheme | 0.01811 | 0.01161 |
| More seeds | 0.01890 | 0.01208 |
| Remove post-normalization | 0.01967 | 0.01255 |
| GRU multiplier ×4 | 0.02059 | 0.01303 |

Source: the results screenshot above. These appear to be successive cumulative
experiments. They show directional agreement, not isolated causal contributions or
proof that the same scores will occur on our historical holdout. The slide describes
120 “purged” dates; the transcript says the intervening period receives online
adaptation. Preserve the label-release protocol instead of treating this as a
blanket prohibition on using released warmup labels.

## Planned comparison and ablations

The independent switches are implemented in `configs/patrick.yaml` and
`experiments/ablations.py`; performance experiments remain outstanding. Run them only
after simulator and causal checks pass, using identical evaluation dates and initial
checkpoints where the intervention allows it. Unrecovered details use the explicit
assumptions in the implementation document.

| Experiment | Vary independently | Keep fixed / validate |
|---|---|---|
| Online adaptation | Updates off/on | Starting checkpoint, stream, seeds; updates only after lag release |
| Seed ensemble | Number of seeds | Seed order fixed in advance; architecture, epochs, training dates, updates |
| Recency weighting | Multiplier off/on | Training partition; d_max from offline training only |
| Full-length weighting | ×1.5 off/on | Separate from recency weighting |
| Post-normalization | Off/on | Assumed final RMSNorm placement; author placement unverified |
| Temporal capacity | GRU multiplier 1/4 | Attention width 64 and other settings |
| Auxiliary targets | responder_6 only/all nine | Loss scaling and target order explicitly specified |
| Fast inference | Separate models/stacked models | Prediction and hidden-state equivalence; runtime and memory |

The existing Grigoreva feature switches (market averages and rolling features) remain
part of the original harness. They are not claimed Patrick Yam features. Distinguish
comparisons matching training history from those preserving each author's choices.

Use pooled weighted zero-mean R² over each 20-day window for our rolling diagnostics,
not an average of daily R². This is our metric-consistent plotting rule; the exact
implementation of the author's plotted rolling score is not available. Report the
overall scored-period R² separately and exclude unscored warmup dates from it.

## Open items

- [ ] Exact numerical columns, category mappings/vocabularies, missing and unknown handling.
- [ ] Gaussian time-transform bounds, endpoint clipping, and day-length handling.
- [ ] Model factory: SiLU/gating wiring, custom GLU and RMSNorm definitions, head activations.
- [ ] Meaning and placement of `use_post_norm`.
- [ ] Target ordering, loss formula, denominator/zero-loss safeguards.
- [ ] How sample multipliers combine with supplied weights; per-asset versus per-day length.
- [ ] Fixed epoch count, batch construction, seed list, and final ensemble composition.
- [ ] Online learning rate, optimizer lifetime across days, and online target/loss choices.
- [ ] Missing-symbol masks, GRU state reset policy, and state behavior after weight updates.
- [ ] Executable training/inference source and mapping to the displayed configuration.
- [ ] Tests: temporal prefix invariance, symbol permutation equivariance, missing-symbol
  handling, stacked/separate inference parity, cached/full-prefix parity, online timing.
- [ ] Shared development/holdout split and compute budget fixed before comparing scores.
- [ ] Resolve the printed 1500–1699 interval against actual dates ending at 1698;
  confirm which scored rows produced 0.02059. See the validation boundary audit.

The separate 120-date validation mention and the 120-day warmup plus about 200-day
scoring scenario must stay distinct. Data actually downloaded spans dates 0–1698;
do not copy a diagram endpoint of 1699 into a split without checking the counts.

## Local status

Competition data is available through `data/competition/train.parquet`: 47,127,338
rows, 1,699 dates, ten partitions. See [verification record](verification.md) and
`artifacts/competition-data.json`. The official gateway matched all 15 synthetic
reference batches. The existing Grigoreva implementation has now completed a
[bounded real-data CUDA pilot](real-data-pilot-report.md), including online-update
and future-label checks. No Patrick Yam model or full-training comparison exists yet.
