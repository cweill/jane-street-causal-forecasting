"""Bounded equivalence/timing benchmark; source research volume is read-only."""

import json
from pathlib import Path

import modal

from scripts.modal_pilot import image

app = modal.App("patrick-replay-acceleration-benchmark")
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
output_volume = modal.Volume.from_name("janestreet-replay-acceleration", create_if_missing=True)


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
        "/accelerated": output_volume,
    },
    include_source=False,
)
def benchmark(source_run: str, benchmark_id: str):
    from dataclasses import replace
    from time import perf_counter

    import polars as pl
    import torch

    from scripts.modal_patrick_reproduction import verified_source
    from src.artifacts import load_predictor, write_json
    from src.data.api_simulator import APISimulator
    from src.data.loader import CachedDaySource, FrameSource
    from src.reproduction import weights_fingerprint
    from src.safety import safety_gate

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path("/research/runs") / source_run
    output = Path("/accelerated/benchmarks") / benchmark_id
    output.mkdir(parents=True, exist_ok=False)
    gate = safety_gate()
    identity = json.loads((root / "identity.json").read_text())
    split = json.loads((root / "splits.json").read_text())
    dates = (split["warmup_dates"] + split["validation_dates"])[:3]
    source = verified_source(identity["dataset_sha256"])
    original_metadata = json.loads((root / "initial_checkpoint/metadata.json").read_text())

    def run(online, fast, data):
        predictor = load_predictor(root / "initial_checkpoint", "cuda")
        predictor.config = replace(predictor.config, enabled=online)
        predictor.fast_inference = fast
        data = CachedDaySource(data) if fast else data
        parts, daily, offset = [], [], 0
        torch.cuda.synchronize()
        start = perf_counter()
        for date in dates:
            day_start = perf_counter()
            result = APISimulator(data, [date], row_offset=offset, prepartition_truth=fast).run(
                predictor.predict
            )
            torch.cuda.synchronize()
            offset += result.predictions.height
            parts.append(result.predictions)
            daily.append(
                {
                    "date": date,
                    "seconds": perf_counter() - day_start,
                    "sse": result.metric.sse,
                    "denominator": result.metric.denominator,
                    "calls": result.calls,
                }
            )
            print(json.dumps({"online": online, "fast": fast, **daily[-1]}), flush=True)
        elapsed = perf_counter() - start
        return predictor, pl.concat(parts), daily, elapsed

    def assert_same(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b, strict=True):
                assert_same(left, right)
        else:
            assert a == b

    results = {}
    try:
        for mode, online in (("offline", False), ("online", True)):
            baseline, p0, days0, seconds0 = run(online, False, source)
            fast, p1, days1, seconds1 = run(online, True, source)
            assert p0.equals(p1), "optimized predictions differ"
            assert baseline.update_log == fast.update_log
            assert_same(baseline.models[0].state_dict(), fast.models[0].state_dict())
            if online:
                assert_same(baseline._optimizers[0].state_dict(), fast._optimizers[0].state_dict())
            for a, b in zip(days0, days1, strict=True):
                assert a["sse"] == b["sse"] and a["denominator"] == b["denominator"]
            results[mode] = {
                "baseline_seconds": seconds0,
                "optimized_seconds": seconds1,
                "speedup": seconds0 / seconds1,
                "baseline_daily": days0,
                "optimized_daily": days1,
                "predictions_scores_weights_optimizer_identical": True,
                "final_weights_sha256": weights_fingerprint(fast.models[0]),
            }
            p1.write_parquet(output / f"{mode}.parquet")
            if online:
                frame = pl.concat([source.day(d) for d in [dates[0] - 1, *dates]])
                frame = frame.with_columns(
                    [
                        pl.when(pl.col("date_id") == dates[-1])
                        .then(999.0)
                        .otherwise(pl.col(f"responder_{i}"))
                        .alias(f"responder_{i}")
                        for i in range(9)
                    ]
                )
                poison, poisoned, _, _ = run(True, True, FrameSource(frame))
                assert p1.equals(poisoned), "future labels changed predictions"
                assert_same(fast.models[0].state_dict(), poison.models[0].state_dict())
                assert_same(fast._optimizers[0].state_dict(), poison._optimizers[0].state_dict())
                results[mode]["future_label_perturbation_identical"] = True
                del poison
            del baseline, fast
        summary = {
            "status": "passed",
            "source_run": source_run,
            "dates": dates,
            "initial_checkpoint_sha256": original_metadata["weights_sha256"],
            "gpu": torch.cuda.get_device_name(),
            "safety_gate": gate,
            "results": results,
            "note": "Implementation benchmark only; no hyperparameter or epoch selection.",
        }
        write_json(output / "result.json", summary)
        return summary
    finally:
        output_volume.commit()
