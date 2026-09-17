"""Run the nine-day correctness pilot in one bounded Modal GPU container.

TMPDIR=/tmp uv run --no-project --python 3.12 --with modal==1.5.5 \
    modal run scripts/modal_pilot.py --data artifacts/pilot-input.parquet
"""

import hashlib
import json
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("janestreet-repro-pilot")
volume = modal.Volume.from_name("janestreet-repro-pilot", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extras=["dev"], frozen=True)
    .env(
        {
            "PYTHONPATH": "/project:/project/scripts",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "4",
            "POLARS_MAX_THREADS": "4",
        }
    )
    .workdir("/project")
)
# Explicit allowlist: no credentials, whole data directory, or personal files.
for directory in ("src", "tests", "scripts", "experiments", "configs"):
    image = image.add_local_dir(
        ROOT / directory, f"/project/{directory}", ignore=["**/__pycache__/**"]
    )
for filename in ("pyproject.toml", "uv.lock"):
    image = image.add_local_file(ROOT / filename, f"/project/{filename}")


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
def run_gpu(input_digest: str, run_id: str):
    from scripts.real_data_pilot import run_pilot
    from src.artifacts import sha256_file

    data = Path("/pilot/inputs") / f"{input_digest}.parquet"
    if sha256_file(data) != input_digest:
        raise ValueError("uploaded pilot input checksum mismatch")
    output = Path("/pilot/runs") / run_id
    try:
        result = run_pilot(data, output, device="cuda")
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output, arcname=run_id)
        return {
            "result": result,
            "archive": f"/runs/{run_id}.tar.gz",
            "archive_sha256": sha256_file(archive),
        }
    finally:
        # Preserve checkpoints/logs even when the pilot raises an exception.
        volume.commit()


@app.local_entrypoint()
def main(data: str):
    data_path = Path(data).resolve()
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = json.loads(data_path.with_suffix(".json").read_text())
    if manifest["sha256"] != digest or manifest["dates"] != list(range(700, 709)):
        raise ValueError("prepare the bounded pilot slice before upload")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    print(f"Run {run_id}: one L4, 20-minute execution timeout, no retries", flush=True)
    # A duplicate content-addressed input may already exist after an interrupted run.
    with volume.batch_upload(force=True) as upload:
        upload.put_file(data_path, f"/inputs/{digest}.parquet")
    result = run_gpu.remote(digest, run_id)
    local = ROOT / "artifacts/modal-pilot"
    local.mkdir(parents=True, exist_ok=True)
    archive_path = local / f"{run_id}.tar.gz"
    with archive_path.open("wb") as handle:
        for block in volume.read_file(result["archive"]):
            handle.write(block)
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != result["archive_sha256"]:
        raise ValueError("downloaded artifact checksum mismatch")
    with tarfile.open(archive_path) as handle:
        handle.extractall(local, filter="data")
    print(f"Downloaded verified run artifacts: {local / run_id}", flush=True)
    print(json.dumps(result["result"], indent=2))
