# Verification record

Verified locally on 2026-09-17 with Python 3.12.12 on macOS ARM64, using `uv.lock`.

| Check | Result |
|---|---|
| `python -m pytest -q` | 61 passed |
| `ruff check .` | Passed |
| `ruff format --check .` | 56 files formatted correctly |
| Actual distributed gateway batching method | All 15 synthetic batches matched |
| Deliberate leakage mutations in disposable copies | All 5 detected by failing tests |
| Six-member Grigoreva and single-seed Patrick online synthetic CLI smoke | Both temporal folds completed for each |
| Reference + five one-switch ablations | All six configurations completed both folds |
| Checkpoint reload, parquet loader, unknown/missing symbols | Covered by tests and CLI replay |

The five negative controls expose a current responder in the callback, deliver same-day
responders at time zero, fit Grigoreva preprocessing outside its declared partition,
mix future features into Patrick’s temporal input, and fit Patrick preprocessing
outside its declared partition. Each caused a
test failure rather than an import or test-collection error. The working source was not
modified during these checks.

Local generated artifacts:

- `artifacts/leakage-mutations.json`: failing-test evidence for each negative control.
- `artifacts/final-online-smoke/`: checkpoint, folds, predictions, and update audits.
- `artifacts/ablation-smoke/comparison.json`: complete synthetic ablation execution results.

Synthetic runs use smaller dimensions, one epoch, and a short rolling window. Their
scores are not evidence of predictive improvements. The published network dimensions
were separately instantiated and exercised in forward-pass tests. No full competition
dataset training, Kaggle RPC submission, or leaderboard optimization was done.
Bounded real-data CUDA runs are recorded below.

Repeat the commands in `README.md` for a fresh verification record after changes.

## Official data download

Downloaded successfully on 2026-09-17 using
`kagglehub.competition_download('jane-street-real-time-market-data-forecasting')`.
The local symlink `data/competition` points to the Kagglehub cache. Pass
`--data data/competition/train.parquet` to the harness, rather than the parent
directory (which also contains example test and lag parquet files).

Metadata checks found 10 training partitions, 47,127,338 rows, 1,699 distinct
dates spanning 0–1698, 79 feature columns, and nine responder columns.
`artifacts/competition-data.json` records the paths, file sizes, schema counts,
date range, and official gateway SHA256.

The official gateway differs from the previously inspected mirror only in import
ordering. After inspecting that diff, its exact hash was added to the verification
allowlist. Running `scripts.verify_gateway` against the downloaded official file
matched all 15 synthetic batches. This is a batching/lag differential check; it does
not establish full RPC or hosted-runtime equivalence. At that initial download
checkpoint, training had not yet started; subsequent pilots are recorded below.

## Real-data CUDA pilot (later on 2026-09-17)

The [Modal L4 pilot](real-data-pilot-report.md) completed one epoch on dates 700–703
and three five-day replays on dates 704–708. Local tests: 49 passed; the remote
causal gate: 42 passed. Unreleased responder perturbations left predictions and
final weights identical. Keyed score recomputation agreed within 1e-12. The run
finished in 320.70 seconds of measured pilot work with 2.09 GiB peak PyTorch CUDA
allocation. Artifacts were downloaded and verified. This supersedes the earlier
statement that no CUDA or real-data training had yet been run; full competition
training and leaderboard optimization still have not been performed.

## Patrick local integration (2026-09-17)

All 61 tests pass, including training-only vocabularies and statistics, temporal
prefix and gradient causality, missing-asset masking, symbol permutation,
full-day versus cached inference, checkpoint replay, delayed online learning,
and independent seed averaging. The Patrick CLI smoke completed both temporal
folds after the mandatory safety gate. Both additional deliberate leakage
mutations were detected. GitHub Actions also validates the shared harness on Linux.

The [implementation decisions](patrick-implementation.md) distinguish recovered
slide settings from assumptions. No reported competition score is reproduced.
