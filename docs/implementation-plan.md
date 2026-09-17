# Jane Street reproduction plan

Goal: implement the requested repository layout with a verified causal simulator first,
then the published GRU and independently configurable ablations.

Architecture: `src/data/api_simulator.py` owns unreleased truth. Predictors only receive
copied test frames and previous-day lag frames. One deterministic feature implementation
is shared by offline preparation and streaming inference. Training partitions and fitted
preprocessors are isolated per fold; validation uses the simulator exclusively.

1. Write failing metric, protocol, lag visibility, and causal perturbation tests.
   Implement strict schemas, day-at-a-time parquet loading, global sufficient-statistic
   scoring, deterministic temporal folds, and API replay. Pass these before model work.
2. Implement source-pinned features and both GRU architectures. Add tests for streaming
   parity, auxiliary label partition boundaries, ragged symbols, hidden state reset,
   delayed online updates, and ensemble independence.
3. Add day-at-a-time offline training, artifact persistence, configs, CLI, and ablation
   orchestration. Use fixed epochs (no validation-dependent training leakage). Every
   experiment run first executes the protocol and causal test suite.
4. Run all tests, negative-control mutations, and small synthetic end-to-end runs. Document
   provenance, deliberate causal corrections, performance limits, and exact commands.

No leaderboard optimization or claims of reproducing leaderboard scores. Competition
data are user-provided; synthetic smoke runs validate execution, not predictive quality.

Completed: all four stages, with the user's requested layout retained. See
`docs/verification.md` for the test, gateway differential, mutation, and smoke-run evidence.
