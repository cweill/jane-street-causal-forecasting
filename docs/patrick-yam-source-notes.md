# Patrick Yam comparison: source notes

Start with the [reproduction tracker](patrick-yam-tracker.md) for the screenshot
archive, reported score table, ablation plan, and open implementation questions.

Sources inspected on 2026-09-17:

- [Kaggle Winners Walkthrough](https://www.youtube.com/watch?v=lfzzPZZyzjE), approximately
  19 minutes 39 seconds. English automatic captions were downloaded successfully with
  yt-dlp. The user also supplied a transcript in the conversation.
- [Author's presentation slides](https://drive.google.com/file/d/1AdzXrOZN699SGHAyb32slLBkEMxBCYiJ/view),
  linked by [the talk organizer](https://www.adml.ai/2025/09/abu-dhabi-machine-learning-season-6-episode-1/).

Local caption artifacts are under `artifacts/references/patrick-yam/`: `transcript.md`
has timestamps; `lfzzPZZyzjE.en-orig.json3` retains the source captions;
`transcript-source.json` records provenance and hashes. These are automatic captions,
not a manually verified transcription. Terms such as GRU, axial attention, einsum,
Optuna, epochs, and seed averaging are frequently misrecognized in the captions.

## Confirmed method details

- Slide 19 confirms 77 numerical inputs clipped to `[-10, 10]`, an identity
  numerical embedding, and three categorical embeddings with cardinalities
  83, 13, and 540, each of width 16. Concatenation therefore has width 125
  (derived), projected to `d_model=64`. There are eight `TransformerLayer2d`
  blocks, followed by an FC head `64 -> 256 -> 128 -> 9`. Input and output
  shapes are `(B, T, A, 77)` and `(B, T, A, 9)`. See the
  [transcribed architecture](patrick-yam-architecture.md).
- Slide 22 confirms a one-layer GRU per block, a projection from
  `rnn_multiplier * d_model` back to `d_model`, three pre-normalized residual
  branches, and a feed-forward activation `GLUact(nn.GELU())` with a halved
  intermediate width. Its constructor defaults are dropout 0.0, temporal
  multiplier 1, and LayerNorm; final constructor overrides are not shown.
  The complete visible block is transcribed in the architecture notes.
- Slide 28's displayed configuration resolves `nheads=8`, `d_hidden=1024`,
  `rnn_multiplier=4` (GRU hidden width 256), dropout 0.0, `norm_type="RMSNorm"`,
  and `activation="silu"`. The last two differ from slide 22's constructor defaults;
  the missing model factory prevents confirming the precise gating wrapper.
  Offline AdamW uses learning rate `5e-4`, weight decay `1e-4`, betas
  `(0.95, 0.9999)`, no scheduler, and gradient accumulation 1. Target weights are
  `[1, 1, 1, 6, 2, 2, 12, 5, 5]`; output ordering still needs code verification.
- Numerical features are standardized using training means and standard deviations,
  then clipped; categorical inputs use embeddings. The numerical and embedded features
  are concatenated and projected into a common hidden dimension.
- One additional time-of-day feature uses a uniform rescaling followed by a Gaussian
  quantile transform. This is not Grigoreva's raw standardized time_id feature.
- A day is represented along time and symbol axes. Repeated blocks mix symbols using
  self-attention, mix time using a GRU, and apply a feed-forward network, with residual
  connections. The slides show pre-normalization. The GRU's hidden width has a separate
  multiplier so temporal capacity can exceed cross-symbol attention capacity.
- Symbol mixing must be permutation equivariant: reordering input symbols must reorder
  the outputs correspondingly. Current-timestamp cross-symbol attention is compatible
  with the competition API, which serves the full timestamp's rows together.
- Multitask supervision uses the nine supplied responders, with larger target weights
  for targets correlated with responder_6. Target weights were explored manually with
  LLM suggestions; exact final weights are not specified in the supplied transcript.
- Per-target score-based losses are normalized by their detached values before combining
  them. This description alone does not establish the exact loss sign, denominator
  safeguards, or zero-loss handling; inspect executable code before reproducing it.
- Training uses all historical dates, with recency weighting
  `(200 + date_id) / (200 + max_date_id)` and a 1.5 multiplier for full 968-step days.
- The local evaluation described uses an offline training interval, then a **120-day
  unscored online-adaptation interval**, followed by about 200 scored days. The printed
  slide endpoints include 1699 despite describing 1699 total dates: use validated
  available-date counts instead of copying an off-by-one endpoint.
- Final offline training uses fixed epochs and averages multiple seeds. The common
  epoch count is selected based on local performance after online adaptation, then
  final models and normalization are fitted on the full available training history.
- PDF page 30 additionally mentions holding out the last **120 dates as validation**
  when discussing CV and epoch selection. This is distinct from the separate
  120-day unscored warmup plus about 200 scored days in the private-leaderboard
  simulation. The slide favors one common epoch for the ensemble over independently
  choosing each seed's best checkpoint; it does not state the chosen epoch count.
  The [validation boundary audit](patrick-yam-validation.md) records the exact
  printed ranges and the unresolved one-day discrepancy with the downloaded data.
- Online learning performs three Adam steps per newly available day, with betas
  `(0.8, 0.95)`. The author reports a fresh optimizer outperforming reuse of the offline
  optimizer, and training on only the latest day outperforming a multi-day replay window.
  This establishes a new optimizer for online learning, not necessarily a reset every
  day. Whether online optimizer state persists across dates remains unknown.
- Inference stacks model weights and uses einsum to evaluate seeds in parallel. GRU
  states are cached between timestamps. The author reports about 2.5 hours for 17 models
  including online learning; this is his reported runtime, not a local measurement.

## Rules for a later comparison

The benchmark's metric and label-release boundary remain fixed. In this harness,
"newly available day" always means the previous day's labels released at the next
day's time zero, even when the talk calls it the current day of training.

Reproduce the author's 120-day warmup as a separately named evaluation scenario. The
existing 200-day warmup reproduces Grigoreva's historical gap experiment; neither is a
universal API constant. For a direct method comparison, run both models on the same
offline training dates, warmup dates, and scored dates. Separate a comparison using the
authors' respective training-history choices from a comparison with matched training data.

All normalization, target-correlation calculations, category-vocabulary construction,
epoch selection, and hyperparameter selection must exclude the final comparison holdout.
Online adaptation may consume holdout labels only after their API release.

Additional causal tests will need to cover masked cross-symbol attention, symbol
permutation equivariance, unknown categories, temporal prefix invariance, and equivalence
between whole-sequence and incremental execution of the complete stacked network.

## Still needed for an exact implementation

The supplied transcript and inspected slides do not specify the complete
numerical/categorical feature lists, mappings for Cat1–Cat3 and unknown-value policy,
time-transform endpoint handling, configuration-to-activation wiring and the custom
GLU implementation, output ordering for the displayed target weights, full loss
implementation, online learning rate, epoch count, or complete training/inference
code. Some may be visible in the video's code/configuration slides.
No complete public training repository has been located yet. Implementing these without
additional evidence would be a documented reconstruction, not an exact reproduction.

No second-place model implementation or changes to the existing benchmark were made
as part of this transcript retrieval.
