"""Detached, resumable Patrick plot run; CPU preparation completes before GPU allocation.

TMPDIR=/tmp uv run --with modal==1.5.5 \\
    modal run --detach scripts/modal_patrick_reproduction.py

Resume with --run-id <same-id> using the same source commit and input data.
"""

import json
import re
import subprocess
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import modal

from scripts.modal_pilot import ROOT, image

app = modal.App("patrick-online-reproduction")
volume = modal.Volume.from_name("janestreet-patrick-reproduction", create_if_missing=True)
MOUNT = Path("/research")
CONFIG = "configs/patrick_ol.yaml"


def verified_source(digest):
    from src.data.loader import ParquetSource
    from src.training.cache import parquet_fingerprint

    path = MOUNT / "datasets" / digest / "train.parquet"
    actual, _ = parquet_fingerprint(path)
    if actual != digest:
        raise ValueError("uploaded dataset checksum mismatch")
    return ParquetSource(path)


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=7200,
    startup_timeout=900,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    volumes={str(MOUNT): volume},
    include_source=False,
)
def prepare_cpu(digest: str, run_id: str, launch: dict):
    from dataclasses import asdict

    from src.artifacts import write_json
    from src.config import load_config
    from src.cv import configured_folds
    from src.data.loader import RestrictedDateSource
    from src.safety import safety_gate
    from src.training.cache import prepare_cached

    volume.reload()
    started = perf_counter()
    gate = safety_gate()
    if gate["code_and_tests_sha256"] != launch["local_safety_code_sha256"]:
        raise ValueError("remote source differs from the locally verified source")
    source = verified_source(digest)
    config = load_config(CONFIG)
    (fold,) = configured_folds(source.dates(), config.cv)
    metadata = MOUNT / "launches" / run_id
    metadata.mkdir(parents=True, exist_ok=True)
    identity = {**launch, "dataset_sha256": digest, "code_sha256": gate["code_and_tests_sha256"]}
    manifest_path = metadata / "launch.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("launch identity changed; restore the recorded source/data to resume")
    write_json(manifest_path, identity)
    write_json(metadata / "splits.json", asdict(fold))
    write_json(metadata / "safety_gate.json", gate)
    archive = metadata / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in (
            "src",
            "tests",
            "scripts",
            "experiments",
            "configs",
            "uv.lock",
            "pyproject.toml",
        ):
            handle.add(
                Path("/project") / name,
                arcname=name,
                filter=lambda info: None if "__pycache__" in info.name else info,
            )
    volume.commit()
    print(
        json.dumps(
            {
                "phase": "cpu_preparation",
                "training_dates": [min(fold.train_dates), max(fold.train_dates)],
            }
        ),
        flush=True,
    )
    try:
        _, cache = prepare_cached(
            RestrictedDateSource(source, fold.train_dates),
            fold.train_dates,
            config.features,
            MOUNT / "preparation_cache",
            digest,
        )
        result = {**cache, "seconds": perf_counter() - started, "status": "prepared"}
        write_json(metadata / "preparation.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=43200,
    startup_timeout=900,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    volumes={str(MOUNT): volume},
    include_source=False,
)
def run_gpu(digest: str, run_id: str):
    import torch

    from src.artifacts import sha256_file, write_json
    from src.config import load_config
    from src.reproduction import run_online_comparison

    volume.reload()
    started = perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source = verified_source(digest)
    output = MOUNT / "runs" / run_id

    def progress(record):
        if record.get("phase") != "training" or record.get("checkpoint_written"):
            print(json.dumps({**record, "wall_seconds": perf_counter() - started}), flush=True)
            volume.commit()

    try:
        result = run_online_comparison(
            source,
            load_config(CONFIG),
            output,
            cache_root=MOUNT / "preparation_cache",
            dataset_sha256=digest,
            resume=output.exists(),
            progress=progress,
        )
        write_json(
            output / "runtime.json",
            {
                "gpu": torch.cuda.get_device_name(),
                "wall_seconds_this_invocation": perf_counter() - started,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        )
        archive = output.with_suffix(".tar.gz")
        # Recovery checkpoints and training cache stay on the Volume. Download the
        # starting model, predictions, daily statistics, plots, and audit manifests.
        with tarfile.open(archive, "w:gz") as handle:

            def compact(info):
                parts = Path(info.name).parts
                return None if "checkpoints" in parts or "training.pt" in parts else info

            handle.add(output, arcname=run_id, filter=compact)
            handle.add(MOUNT / "launches" / run_id, arcname=f"{run_id}/launch")
        summary = {
            "status": result["status"],
            "archive": f"/runs/{run_id}.tar.gz",
            "sha256": sha256_file(archive),
            "offline_r2": result["offline"]["score"],
            "online_r2": result["online"]["score"],
        }
        write_json(output / "download.json", summary)
        return summary
    except Exception as error:
        if output.exists():
            write_json(
                output / "failure.json", {"type": type(error).__name__, "message": str(error)}
            )
        raise
    finally:
        volume.commit()


@app.function(
    image=image,
    cpu=(0.125, 0.125),
    memory=(512, 512),
    timeout=54000,
    startup_timeout=900,
    max_containers=1,
    retries=0,
    include_source=False,
)
def orchestrate(digest: str, run_id: str, launch: dict):
    # Keep both stages server-side so disconnecting the local terminal during CPU
    # preparation cannot prevent the later GPU stage from being scheduled.
    prepare_cpu.remote(digest, run_id, launch)
    return run_gpu.remote(digest, run_id)


@app.local_entrypoint()
def main(data: str = "data/competition/train.parquet", run_id: str = ""):
    from src.artifacts import sha256_file
    from src.config import load_config
    from src.cv import configured_folds
    from src.data.loader import ParquetSource
    from src.safety import safety_gate
    from src.training.cache import parquet_fingerprint

    gate = safety_gate()
    data_path = Path(data).resolve()
    if not data_path.is_dir():
        raise ValueError("the complete partitioned competition training directory is required")
    config = load_config(CONFIG)
    configured_folds(ParquetSource(data_path).dates(), config.cv)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit the source before starting or resuming the long run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run ID must be a simple directory name")
    print(f"Hashing input for {run_id}; GPU has not been allocated.", flush=True)
    digest, files = parquet_fingerprint(data_path)
    launch = {
        "run_id": run_id,
        "git_commit": commit,
        "config": config.to_dict(),
        "local_safety_code_sha256": gate["code_and_tests_sha256"],
        "files": files,
        "bounds": {
            "cpu_preparation_seconds": 7200,
            "gpu_seconds": 43200,
            "gpu": "L4",
            "retries": 0,
        },
    }
    local = ROOT / "artifacts/patrick-reproduction" / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "launch.json").write_text(
        json.dumps({**launch, "dataset_sha256": digest}, indent=2) + "\n"
    )
    print(
        f"Uploading {sum(f['bytes'] for f in files) / 2**30:.2f} GiB to the persistent research Volume.",
        flush=True,
    )
    with volume.batch_upload(force=True) as upload:
        for file in files:
            upload.put_file(
                data_path / file["name"], f"/datasets/{digest}/train.parquet/{file['name']}"
            )
    print(
        f"Starting {run_id}; CPU preparation then one L4 with a 12-hour execution cap.", flush=True
    )
    result = orchestrate.remote(digest, run_id, launch)
    archive = local.parent / f"{run_id}.tar.gz"
    with archive.open("wb") as handle:
        for chunk in volume.read_file(result["archive"]):
            handle.write(chunk)
    if sha256_file(archive) != result["sha256"]:
        raise ValueError("downloaded result checksum mismatch")
    with tarfile.open(archive) as handle:
        handle.extractall(local.parent, filter="data")
    print(json.dumps({"artifacts": str(local), **result}, indent=2), flush=True)
