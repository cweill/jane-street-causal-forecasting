"""Bounded L4 replay benchmark against the saved, unoptimized Patrick pilot."""

import json
import tarfile
import uuid
from pathlib import Path
from time import perf_counter

import modal

from scripts.modal_pilot import ROOT, image, volume

app = modal.App("patrick-cache-benchmark")


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
    volumes={"/pilot": volume},
    include_source=False,
)
def benchmark(source_run: str, run_id: str):
    import numpy as np
    import polars as pl
    import torch

    from scripts.real_data_pilot import replay
    from src.artifacts import sha256_file, write_json
    from src.config import load_config
    from src.data.loader import FrameSource
    from src.safety import safety_gate
    from src.training.cache import prepare_cached

    gate = safety_gate()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    baseline = Path("/pilot/runs") / source_run
    output = Path("/pilot/runs") / run_id
    output.mkdir()
    old = json.loads((baseline / "result.json").read_text())
    digest = json.loads((baseline / "data_manifest.json").read_text())["sha256"]
    data = Path("/pilot/inputs") / f"{digest}.parquet"
    assert sha256_file(data) == digest
    frame = pl.read_parquet(data)
    config = load_config("configs/patrick.yaml")
    try:
        started = perf_counter()
        first, cold = prepare_cached(
            FrameSource(frame),
            tuple(range(700, 704)),
            config.features,
            output / "preparation_cache",
            digest,
        )
        cold_seconds = perf_counter() - started
        started = perf_counter()
        _, warm = prepare_cached(
            FrameSource(frame),
            tuple(range(700, 704)),
            config.features,
            output / "preparation_cache",
            digest,
        )
        warm_seconds = perf_counter() - started
        assert not cold["hit"] and warm["hit"]
        for date in range(700, 704):
            with np.load(baseline / "training_cache" / f"{date}.npz") as previous:
                for name, current in zip(
                    ("x", "cats", "mask", "y", "w"), first.read(date), strict=True
                ):
                    np.testing.assert_array_equal(previous[name], current)
        results = {}
        for mode, online in (("offline", False), ("online", True)):
            # Load the original checkpoint AND original optimizer settings, including
            # lr=0.0003, solely for exact before/after implementation equivalence.
            result = replay(frame, baseline / "checkpoint", output / mode, "cuda", online)
            for date in range(704, 709):
                name = f"date_{date}.parquet"
                assert pl.read_parquet(output / mode / "predictions" / name).equals(
                    pl.read_parquet(baseline / mode / "predictions" / name)
                ), f"prediction drift: {mode}/{date}"
            assert result["final_weights_sha256"] == old[mode]["final_weights_sha256"]
            results[mode] = {
                "seconds": result["seconds"],
                "baseline_seconds": old[mode]["seconds"],
                "speedup": old[mode]["seconds"] / result["seconds"],
                "predictions_and_final_weights_identical": True,
            }
        summary = {
            "status": "passed",
            "source_run": source_run,
            "gpu": torch.cuda.get_device_name(),
            "cold_preparation_seconds": cold_seconds,
            "warm_cache_seconds": warm_seconds,
            "prepared_arrays_identical": True,
            "replays": results,
            "safety_gate": gate,
        }
        write_json(output / "result.json", summary)
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            # Prepared arrays remain on the Volume; download the useful compact evidence.
            for name in ("result.json", "offline", "online"):
                handle.add(output / name, arcname=f"{run_id}/{name}")
        return {
            "summary": summary,
            "archive": f"/runs/{run_id}.tar.gz",
            "sha256": sha256_file(archive),
        }
    finally:
        volume.commit()


@app.local_entrypoint()
def main(source_run: str = "20260917T232828Z-0b458313"):
    import hashlib

    run_id = "cache-benchmark-" + uuid.uuid4().hex[:10]
    result = benchmark.remote(source_run, run_id)
    destination = ROOT / "artifacts/modal-pilot"
    archive = destination / f"{run_id}.tar.gz"
    with archive.open("wb") as handle:
        for chunk in volume.read_file(result["archive"]):
            handle.write(chunk)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["sha256"]
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    print(json.dumps({"artifacts": str(destination / run_id), **result["summary"]}, indent=2))
