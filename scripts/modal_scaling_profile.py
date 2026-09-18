"""Bounded L4 phase/scaling measurements, isolated from active research runs."""

import json
from pathlib import Path

import modal

from scripts.modal_pilot import image

app = modal.App("patrick-training-scaling-profile")
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
output_volume = modal.Volume.from_name("janestreet-replay-acceleration")


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=1200,
    startup_timeout=900,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    volumes={
        "/research": source_volume.with_mount_options(read_only=True),
        "/profiles": output_volume,
    },
    include_source=False,
)
def profile_scaling(source_run: str, profile_id: str):
    import copy
    import gzip
    import statistics
    from dataclasses import replace
    from time import perf_counter
    from unittest.mock import patch

    import polars as pl
    import torch

    from scripts.modal_patrick_reproduction import verified_source
    from scripts.profile_patrick_scaling import synchronize, timed_replay, training_ranges
    from src.artifacts import load_predictor, write_json
    from src.data.api_simulator import APISimulator
    from src.data.loader import FrameSource
    from src.safety import safety_gate
    from src.training.patrick import PatrickPredictor

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = True  # Match the existing training policy.
    root = Path("/research/runs") / source_run
    output = Path("/profiles/scaling-profiles") / profile_id
    output.mkdir(parents=True, exist_ok=False)
    gate = safety_gate()
    print("Safety gate passed; verifying the immutable source", flush=True)
    identity = json.loads((root / "identity.json").read_text())
    started = perf_counter()
    parquet = verified_source(identity["dataset_sha256"])
    source = FrameSource(pl.concat([parquet.day(d) for d in (1379, 1380, 1381, 1382)]))
    source_seconds = perf_counter() - started
    base = load_predictor(root / "initial_checkpoint", "cuda")

    def make_predictor(count):
        models = [copy.deepcopy(base.models[0]) for _ in range(count)]
        with torch.no_grad():
            for i, model in enumerate(models):
                if i:
                    torch.manual_seed(8000 + i)
                    for p in model.parameters():
                        p.add_(0.001 * torch.randn_like(p))
                for block in model.blocks:
                    block.gru.flatten_parameters()
        return PatrickPredictor(
            models,
            copy.deepcopy(base.features),
            replace(base.config, enabled=True),
            seeds=tuple(range(count)),
            stacked_inference=True,
        )

    def assert_equal(a, b):
        assert a.update_log == b.update_log
        for x, y in zip(a.models, b.models, strict=True):
            for key, value in x.state_dict().items():
                torch.testing.assert_close(value, y.state_dict()[key], rtol=0, atol=0)
        for x, y in zip(a._optimizers, b._optimizers, strict=True):
            for key, state in x.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value, y.state_dict()["state"][key][name], rtol=0, atol=0
                    )

    records = []
    try:
        for count in (1, 4, 17):
            predictor = make_predictor(count)
            torch.cuda.reset_peak_memory_stats()
            daily, predictions = timed_replay(predictor, source, [1380, 1381, 1382])
            steady = daily[1:]  # Initial day has no eligible online update.
            summary = {
                "models": count,
                "daily": daily,
                "median": {
                    name: statistics.median(d[name] for d in steady)
                    for name in (
                        "total_seconds",
                        "update_seconds",
                        "panel_seconds",
                        "stack_seconds",
                        "inference_seconds",
                        "feature_seconds",
                        "other_seconds",
                    )
                },
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            update_seconds = summary["median"]["update_seconds"]
            summary.update(
                model_days_updated_per_second=count / update_seconds,
                optimizer_steps_per_second=count * predictor.config.steps / update_seconds,
                update_seconds_per_model=update_seconds / count,
                update_fraction=update_seconds / summary["median"]["total_seconds"],
            )
            if count == 1:
                reference = make_predictor(1)
                expected = APISimulator(source, [1380, 1381, 1382], prepartition_truth=True).run(
                    reference.predict
                )
                assert pl.concat(predictions).equals(expected.predictions)
                assert_equal(predictor, reference)
                summary["timing_preserves_predictions_weights_adam_exactly"] = True
                del reference
            records.append(summary)
            write_json(output / f"models_{count}.json", summary)
            output_volume.commit()
            print(json.dumps(summary), flush=True)
            del predictor, predictions
            torch.cuda.empty_cache()

        # An independent detailed trace profiles the actual one-model update.
        # It is excluded from the throughput table because profiling adds overhead.
        predictor = make_predictor(1)
        reference = make_predictor(1)
        APISimulator(source, [1380], prepartition_truth=True).run(predictor.predict)
        APISimulator(source, [1380], prepartition_truth=True).run(reference.predict)
        original_update = predictor._update
        trace_result = {}

        def traced_update(*args, **kwargs):
            synchronize(predictor)
            with (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as trace,
                training_ranges(),
                torch.profiler.record_function("phase/update"),
            ):
                started = perf_counter()
                result = original_update(*args, **kwargs)
                synchronize(predictor)
                trace_result["wall_seconds"] = perf_counter() - started
            events = trace.key_averages()

            def event_record(e):
                return {
                    "name": e.key,
                    "calls": e.count,
                    "cpu_total_seconds": e.cpu_time_total / 1e6,
                    "cpu_self_seconds": e.self_cpu_time_total / 1e6,
                    "gpu_total_seconds": e.device_time_total / 1e6,
                    "gpu_self_seconds": e.self_device_time_total / 1e6,
                }

            trace_result["phases"] = [event_record(e) for e in events if e.key.startswith("phase/")]
            trace_result["top_cpu_operators"] = [
                event_record(e)
                for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
            ]
            trace_result["top_gpu_operators"] = [
                event_record(e)
                for e in sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:15]
            ]
            path = output / "online_update_trace.json"
            trace.export_chrome_trace(str(path))
            data = json.loads(path.read_text())
            intervals = sorted(
                (e["ts"], e["ts"] + e["dur"])
                for e in data["traceEvents"]
                if e.get("cat") == "kernel" and "dur" in e
            )
            if not intervals:
                raise RuntimeError("CUDA kernel trace unavailable; cannot infer GPU activity")
            covered, end = 0.0, float("-inf")
            for start, stop in intervals:
                covered += max(0.0, stop - max(start, end))
                end = max(end, stop)
            trace_result["gpu_kernel_active_seconds"] = covered / 1e6
            trace_result["gpu_kernel_active_fraction_of_profiled_wall"] = (
                covered / 1e6 / trace_result["wall_seconds"]
            )
            with gzip.open(str(path) + ".gz", "wb") as compressed:
                compressed.write(path.read_bytes())
            path.unlink()
            return result

        with patch.object(predictor, "_update", traced_update):
            actual = APISimulator(source, [1381], prepartition_truth=True).run(predictor.predict)
        expected = APISimulator(source, [1381], prepartition_truth=True).run(reference.predict)
        assert actual.predictions.equals(expected.predictions)
        assert_equal(predictor, reference)
        trace_result["instrumentation_preserves_predictions_weights_adam_exactly"] = True
        result = {
            "status": "passed",
            "source_run": source_run,
            "gpu": torch.cuda.get_device_name(),
            "safety_gate": gate,
            "source_verification_and_read_seconds": source_seconds,
            "precision": {"matmul": "highest", "cudnn_allow_tf32": True},
            "online_steps": base.config.steps,
            "online_learning_rate": base.config.learning_rate,
            "scaling": records,
            "trace": trace_result,
            "scope": "Three real replay days, two eligible online updates per ensemble. Distinct perturbed checkpoint copies, not trained seeds. Resident source data; excludes storage/checkpoint/W&B work. Trace overhead excluded from scaling table.",
        }
        write_json(output / "result.json", result)
        print(json.dumps({"status": "passed", "trace": trace_result}), flush=True)
        return result
    finally:
        output_volume.commit()
