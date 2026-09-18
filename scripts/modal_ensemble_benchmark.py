"""Bounded L4 benchmark of stacked inference; never writes to ongoing replay runs."""

import json
from pathlib import Path

import modal

from scripts.modal_pilot import image

app = modal.App("patrick-stacked-ensemble-benchmark")
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
output_volume = modal.Volume.from_name("janestreet-replay-acceleration")


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=1200,
    startup_timeout=900,
    min_containers=0,
    max_containers=1,
    retries=0,
    volumes={
        "/research": source_volume.with_mount_options(read_only=True),
        "/benchmarks": output_volume,
    },
    include_source=False,
)
def benchmark(source_run: str, benchmark_id: str):
    import copy
    import statistics
    from dataclasses import replace
    from time import perf_counter

    import numpy as np
    import polars as pl
    import torch

    from scripts.modal_patrick_reproduction import verified_source
    from src.artifacts import load_predictor, write_json
    from src.data.api_simulator import APISimulator
    from src.data.loader import FrameSource
    from src.data.schema import test_view
    from src.models.patrick_ensemble import StackedPatrick
    from src.safety import safety_gate
    from src.training.patrick import PatrickPredictor

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    root = Path("/research/runs") / source_run
    output = Path("/benchmarks/ensemble-benchmarks") / benchmark_id
    output.mkdir(parents=True, exist_ok=False)
    gate = safety_gate()
    base = load_predictor(root / "initial_checkpoint", "cuda")
    identity = json.loads((root / "identity.json").read_text())
    source = verified_source(identity["dataset_sha256"])
    frame = pl.concat([source.day(d) for d in (1379, 1380, 1381)])
    batches = []
    public = test_view(frame.filter(pl.col("date_id") == 1380))
    for batch in public.partition_by("time_id", maintain_order=True)[:128]:
        x, cats = base.features.transform(batch)
        batches.append((torch.as_tensor(x, device="cuda"), torch.as_tensor(cats, device="cuda")))
    # Model-only throughput uses a fixed symbol set. Full API replay below covers
    # dynamic symbols and performs actual daily updates using released labels.
    assets = min(x.shape[0] for x, _ in batches)
    batches = [(x[:assets], cats[:assets]) for x, cats in batches]

    def models_for(count):
        models = [copy.deepcopy(base.models[0]) for _ in range(count)]
        with torch.no_grad():
            for i, model in enumerate(models[1:], start=1):
                torch.manual_seed(8000 + i)
                for parameter in model.parameters():
                    parameter.add_(0.001 * torch.randn_like(parameter))
        return models

    def assert_optimizer_equal(a, b):
        for x, y in zip(a._optimizers, b._optimizers, strict=True):
            for key, state in x.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value, y.state_dict()["state"][key][name], rtol=0, atol=0
                    )

    timings = []
    try:
        for count in (1, 4, 17):
            models = models_for(count)
            torch.cuda.synchronize()
            started = perf_counter()
            engine = StackedPatrick(models)
            torch.cuda.synchronize()
            stack_seconds = perf_counter() - started

            def measure(stacked, *, count=count, models=models, engine=engine):
                hidden = None if stacked else [None] * count
                outputs = []
                torch.cuda.synchronize()
                start = perf_counter()
                with torch.no_grad():
                    for x, cats in batches:
                        if stacked:
                            prediction, hidden = engine.forward_step(x, cats, hidden)
                        else:
                            members = []
                            for i, model in enumerate(models):
                                prediction, hidden[i] = model.forward_step(
                                    x[None, None], cats[None, None], hidden[i]
                                )
                                members.append(prediction[0, 0])
                            prediction = torch.stack(members)
                        outputs.append(prediction)
                torch.cuda.synchronize()
                return torch.stack(outputs), hidden, perf_counter() - start

            # Warm both implementations, then alternate timing order twice.
            measure(False)
            measure(True)
            times = {False: [], True: []}
            max_error, state_error = 0.0, 0.0
            for order in ((False, True), (True, False)):
                samples = {flag: measure(flag) for flag in order}
                reference, reference_states, slow = samples[False]
                actual, states, fast = samples[True]
                torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
                expected_states = torch.stack([torch.cat(s) for s in reference_states])
                torch.testing.assert_close(states, expected_states, rtol=2e-5, atol=2e-6)
                max_error = max(max_error, (actual - reference).abs().max().item())
                state_error = max(state_error, (states - expected_states).abs().max().item())
                times[False].append(slow)
                times[True].append(fast)
            slow, fast = statistics.median(times[False]), statistics.median(times[True])
            record = {
                "models": count,
                "timestamps": len(batches),
                "assets": assets,
                "loop_seconds": slow,
                "stacked_seconds": fast,
                "speedup": slow / fast,
                "stack_seconds": stack_seconds,
                "max_abs_prediction_error": max_error,
                "max_abs_hidden_error": state_error,
            }
            timings.append(record)
            print(json.dumps(record), flush=True)
            del models, engine, measure

        models = models_for(3)

        def replay(stacked, data):
            predictor = PatrickPredictor(
                copy.deepcopy(models),
                copy.deepcopy(base.features),
                replace(base.config, enabled=True),
                seeds=(0, 1, 2),
                stacked_inference=stacked,
            )
            predictor.fast_inference = True
            torch.cuda.synchronize()
            start = perf_counter()
            result = APISimulator(FrameSource(data), [1380, 1381], prepartition_truth=True).run(
                predictor.predict
            )
            torch.cuda.synchronize()
            return predictor, result, perf_counter() - start

        slow, a, slow_seconds = replay(False, frame)
        fast, b, fast_seconds = replay(True, frame)
        np.testing.assert_allclose(
            a.predictions["responder_6"], b.predictions["responder_6"], rtol=2e-5, atol=2e-6
        )
        assert slow.update_log == fast.update_log
        for x, y in zip(slow.models, fast.models, strict=True):
            for key, value in x.state_dict().items():
                torch.testing.assert_close(value, y.state_dict()[key], rtol=0, atol=0)
        assert_optimizer_equal(slow, fast)
        poisoned = frame.with_columns(
            [
                pl.when(pl.col("date_id") == 1381)
                .then(999.0)
                .otherwise(pl.col(f"responder_{i}"))
                .alias(f"responder_{i}")
                for i in range(9)
            ]
        )
        altered, c, _ = replay(True, poisoned)
        assert b.predictions.equals(c.predictions)
        for x, y in zip(fast.models, altered.models, strict=True):
            for key, value in x.state_dict().items():
                torch.testing.assert_close(value, y.state_dict()[key], rtol=0, atol=0)
        assert_optimizer_equal(fast, altered)
        summary = {
            "status": "passed",
            "source_run": source_run,
            "gpu": torch.cuda.get_device_name(),
            "safety_gate": gate,
            "timings": timings,
            "online_replay": {
                "models": 3,
                "dates": [1380, 1381],
                "loop_seconds": slow_seconds,
                "stacked_seconds": fast_seconds,
                "speedup": slow_seconds / fast_seconds,
                "max_abs_prediction_error": float(
                    np.max(
                        np.abs(
                            a.predictions["responder_6"].to_numpy()
                            - b.predictions["responder_6"].to_numpy()
                        )
                    )
                ),
                "absolute_r2_difference": abs(a.metric.score - b.metric.score),
                "weights_and_optimizer_identical": True,
                "future_labels_no_effect": True,
            },
            "atol": 2e-6,
            "rtol": 2e-5,
            "note": "Distinct perturbed copies of one checkpoint for correctness/throughput only; not trained seed ensembles or a score experiment.",
        }
        write_json(output / "result.json", summary)
        return summary
    finally:
        output_volume.commit()
