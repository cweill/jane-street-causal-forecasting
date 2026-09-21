# Patrick reconstruction: bounded Modal L4 pilot

Completed 2026-09-17. **Passed.** The [Modal App](https://modal.com/apps/cweill/main/ap-my1K6wKtmYjA03KBIXHfvJ)
stopped normally and a subsequent status check showed zero running tasks.
No persistent deployment or larger training job was started.

The [machine-readable record](references/patrick-pilot.json) preserves configuration,
splits, input/source hashes, scores, timing, update audits, and verification results.
Local artifacts are in `artifacts/modal-pilot/20260917T232828Z-0b458313/`;
the Modal Volume `janestreet-repro-pilot` retains `/runs/20260917T232828Z-0b458313/`.
The downloaded archive passed its SHA256 check. Independent verification was
performed locally after download and saved alongside the artifacts.

## Fixed workload

The pilot uses a 251,680-row slice: train 700–703, unscored warmup 704–705,
score 706–708. Dates 1499–1698 were excluded. One seed, one fixed epoch, and full
published dimensions: eight blocks, width 64, eight attention heads, temporal GRU
width 256, FFN width 1024, and head 64→256→128→9. The three categorical embeddings
have width 16. All nine responder targets are enabled.

Preprocessing and training choices are listed in the
[reconstruction assumptions](patrick-implementation.md). In particular, the
online learning rate of 0.0003 and persistence of online Adam moments across days
are explicit choices; neither was established by the source material.

Each of three replays made 4,840 API-style calls. Frozen and online runs started
from the same saved checkpoint. The third run repeated online replay with every
date-708 responder changed. Only the final three days contributed to the metric:
84,216 scored rows per normal replay.

## Verification

| Check | Result |
|---|---|
| Local full suite / remote pre-training safety gate | 61 / 53 tests passed |
| Deliberate leakage mutations | All five detected before this pilot |
| CUDA deterministic training and streaming inference | Completed |
| Lag availability | Full previous-day responders at time zero only; exact keyed values checked |
| Online updates | Three Adam steps at each of (705,0), (706,0), (707,0), (708,0), using dates 704–707 |
| Frozen normalization | Mean and scale unchanged during replay |
| Initial model and first warmup predictions | Identical with online updates on/off |
| Frozen model weights | Initial and final hashes identical |
| Unreleased responder perturbation | Every prediction and final weights identical to normal online replay |
| Independent keyed score recomputation | Absolute errors below 7e-16 |
| Artifact download | Archive SHA256 verified |

The first replay receives date-703 labels but has no matching cached replay inputs,
so the first update is at date 705. Synthetic tests additionally cover irregular
symbol availability, row permutation, future-feature gradients, full-day versus
cached inference, train-only vocabularies, checkpoint reload, and independent seeds.
These tests cover the implemented boundary cases; they do not prove immunity to
arbitrary future code changes or establish full Kaggle RPC/runtime equivalence.

## Runtime and memory

One NVIDIA L4, four CPU cores, 16 GiB host allocation. Python 3.12,
torch 2.14.0, polars 1.44.2, numpy 2.5.3, frozen uv.lock.

| Measurement | Value |
|---|---:|
| Training feature preparation | 7.37 s |
| Four daily training batches / one epoch | 3.44 s |
| Frozen replay, five days | 137.72 s |
| Online replay, five days | 151.79 s |
| Complete measured pilot, including third replay and checks | 450.81 s |
| Host process peak RSS | 5.30 GiB |
| Peak PyTorch CUDA allocation | 4.29 GiB |
| Frozen call median / p95 / maximum | 25.37 / 27.52 / 64.87 ms |
| Online call median / p95 / maximum | 25.80 / 28.27 / 3182.86 ms |

Call times include feature construction and any daily update and synchronize CUDA.
The complete-work timer excludes the remote test gate, image build, startup,
upload, and archive download. GPU allocation excludes non-PyTorch device usage.
Actual account charges were not queried. These timings are for one seed; the
source's stacked/einsum multi-seed inference optimization remains unimplemented.

## Diagnostic scores

| Replay | Weighted zero-mean R² |
|---|---:|
| Online off | -0.004389643758 |
| Online on | 0.003859549438 |

This is a correctness pilot on four training days and three scored days. It does
not reproduce Patrick’s reported 0.02059.
A predictive comparison still requires a shared development protocol, fixed training
budgets, and a final holdout used only after choices are frozen. No leaderboard
optimization or final-holdout training/evaluation was performed.
