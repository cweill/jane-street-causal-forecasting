"""Deploy independent replay workers; no training or writes to the source volume.

The separate CPU observer uses per-mode date axes so out-of-order completion of
the two jobs cannot drop metrics from the slower job.
"""

import hashlib
import json
from pathlib import Path

import modal

from scripts.modal_pilot import ROOT, image

APP_NAME = "patrick-parallel-replay"
app = modal.App(APP_NAME)
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
output_volume = modal.Volume.from_name("janestreet-replay-acceleration", create_if_missing=True)
tracking_volume = modal.Volume.from_name("janestreet-wandb-monitor", create_if_missing=True)
MOUNT = Path("/accelerated")
KERNEL_FILES = (
    "src/models/patrick_yam.py",
    "src/training/patrick.py",
    "src/data/api_simulator.py",
    "src/data/loader.py",
    "src/data/patrick_features.py",
    "src/data/normalization.py",
    "src/data/schema.py",
    "uv.lock",
    "pyproject.toml",
)


def kernel_fingerprint():
    digest = hashlib.sha256()
    for name in KERNEL_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


tracking_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extras=["dev"], frozen=True)
    .uv_pip_install("wandb==0.30.0")
    .env(
        {"PYTHONPATH": "/project:/project/scripts", "OMP_NUM_THREADS": "1", "WANDB_CONSOLE": "off"}
    )
    .workdir("/project")
)
for directory in ("src", "scripts", "tests", "experiments", "configs"):
    tracking_image = tracking_image.add_local_dir(
        ROOT / directory, f"/project/{directory}", ignore=["**/__pycache__/**"]
    )
for name in ("pyproject.toml", "uv.lock"):
    tracking_image = tracking_image.add_local_file(ROOT / name, f"/project/{name}")


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=300,
    retries=0,
    include_source=False,
    volumes={
        "/research": source_volume.with_mount_options(read_only=True),
        "/accelerated": output_volume,
    },
)
def prepare(run_id: str, launch: dict):
    from src.artifacts import write_json
    from src.safety import safety_gate

    source_volume.reload()
    output_volume.reload()
    root = Path("/research/runs") / launch["source_run"]
    benchmark = json.loads(
        (MOUNT / "benchmarks" / launch["benchmark_id"] / "result.json").read_text()
    )
    metadata = json.loads((root / "initial_checkpoint/metadata.json").read_text())
    if (
        benchmark["status"] != "passed"
        or benchmark["source_run"] != launch["source_run"]
        or benchmark["initial_checkpoint_sha256"] != metadata["weights_sha256"]
        or benchmark["safety_gate"]["code_and_tests_sha256"] != launch["benchmark_code_sha256"]
        or kernel_fingerprint() != launch["benchmark_kernel_sha256"]
    ):
        raise ValueError("benchmark/checkpoint/implementation identity mismatch")
    gate = safety_gate()
    split = json.loads((root / "splits.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    record = {
        **launch,
        "code_sha256": gate["code_and_tests_sha256"],
        "source_identity": identity,
        "splits": split,
        "initial_checkpoint_sha256": metadata["weights_sha256"],
    }
    output = MOUNT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", record)
    write_json(output / "safety_gate.json", gate)
    output_volume.commit()
    return record


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=14400,
    startup_timeout=900,
    min_containers=0,
    max_containers=2,
    retries=0,
    volumes={
        "/research": source_volume.with_mount_options(read_only=True),
        "/accelerated": output_volume,
    },
    include_source=False,
)
def replay(run_id: str, mode: str):
    import torch

    from scripts.modal_patrick_reproduction import verified_source
    from src.parallel_replay import run_replay_mode

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source_volume.reload()
    output_volume.reload()
    root = MOUNT / "runs" / run_id
    launch = json.loads((root / "launch.json").read_text())
    split = launch["splits"]
    source = verified_source(launch["source_identity"]["dataset_sha256"])

    def progress(record):
        print(json.dumps(record), flush=True)
        # One durable commit per completed day, rather than also at its start.
        if "completed_date" in record or record.get("complete"):
            output_volume.commit()

    try:
        return run_replay_mode(
            source,
            root / mode,
            checkpoint=Path("/research/runs") / launch["source_run"] / "initial_checkpoint",
            mode=mode,
            replay_dates=split["warmup_dates"] + split["validation_dates"],
            scored_dates=split["validation_dates"],
            device="cuda",
            provenance=launch,
            fast=True,
            progress=progress,
            source_prefix=Path("/research/runs") / launch["source_run"] / "offline"
            if mode == "offline"
            else None,
        )
    finally:
        output_volume.commit()


@app.function(
    image=tracking_image,
    cpu=1,
    memory=2048,
    timeout=18000,
    startup_timeout=900,
    min_containers=0,
    max_containers=1,
    retries=0,
    include_source=False,
    volumes={"/accelerated": output_volume, "/tracking": tracking_volume},
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
)
def observe(run_id: str, calls: dict, entity="cweill-self", project="janestreet-repro"):
    import time
    from datetime import UTC, datetime

    import wandb

    from src.monitoring import evaluation_events
    from src.parallel_replay import finish_comparison

    output_volume.reload()
    root = MOUNT / "runs" / run_id
    launch = json.loads((root / "launch.json").read_text())
    dates = launch["splits"]["warmup_dates"] + launch["splits"]["validation_dates"]
    tracking_directory = Path("/tracking") / run_id
    tracking_directory.mkdir(parents=True, exist_ok=True)
    tracker = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=f"Patrick parallel replay · {run_id}",
        group=launch["source_run"],
        job_type="parallel-replay-monitor",
        resume="allow",
        dir=str(tracking_directory),
        save_code=False,
        config={"launch": launch, "calls": calls},
        settings=wandb.Settings(
            x_disable_stats=True, disable_git=True, disable_code=True, console="off"
        ),
    )
    for mode in calls:
        tracker.define_metric(f"{mode}/*", step_metric=f"{mode}/date_id")
    seen = {mode: int(tracker.summary.get(f"{mode}/days_completed", 0)) for mode in calls}
    tracker.summary["monitor_status"] = "running"
    tracker.summary["source_training_run"] = launch["source_run"]
    started = time.monotonic()
    try:
        while time.monotonic() - started < 17000:
            output_volume.reload()
            complete = True
            for mode, call_id in calls.items():
                for event in evaluation_events(root, mode, replay_dates=dates, total_batches=0):
                    metrics = event["metrics"]
                    count = metrics[f"{mode}/days_completed"]
                    if count > seen[mode]:
                        metrics[f"{mode}/date_id"] = metrics.pop("eval/date_id")
                        tracker.log(metrics, commit=True)
                        seen[mode] = count
                try:
                    modal.FunctionCall.from_id(call_id).get(timeout=0)
                except TimeoutError:
                    complete = False
            tracker.summary["observed_at_utc"] = datetime.now(UTC).isoformat()
            if complete:
                output_volume.reload()
                # Flush any final committed days before emitting the comparison image.
                for mode in calls:
                    for event in evaluation_events(root, mode, replay_dates=dates, total_batches=0):
                        metrics = event["metrics"]
                        if metrics[f"{mode}/days_completed"] > seen[mode]:
                            metrics[f"{mode}/date_id"] = metrics.pop("eval/date_id")
                            tracker.log(metrics, commit=True)
                            seen[mode] = metrics[f"{mode}/days_completed"]
                summary = finish_comparison(
                    root, scored_start=launch["splits"]["validation_dates"][0]
                )
                for mode in calls:
                    tracker.summary[f"{mode}/final_scored_r2"] = summary[mode]["score"]
                tracker.log(
                    {"plots/online_learning": wandb.Image(str(root / "online_learning.png"))}
                )
                tracker.summary["monitor_status"] = "complete"
                tracker.finish()
                return {"status": "complete", "wandb_url": tracker.url}
            output_volume.commit()
            time.sleep(30)
        raise TimeoutError("observer time limit reached; GPU workers remain independent")
    except Exception as error:
        tracker.summary["monitor_status"] = "error; check independent GPU calls"
        tracker.summary["observer_error_type"] = type(error).__name__
        tracker.finish(exit_code=1)
        raise
    finally:
        output_volume.commit()
        tracking_volume.commit()


def main():
    import argparse
    import subprocess
    import uuid
    from datetime import UTC, datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit the verified implementation before submitting")
    # Verify the benchmark really ran the declared source tree, and require the
    # optimized kernels to be unchanged since that measured run.
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", args.benchmark_commit], cwd=ROOT, text=True
    ).splitlines()
    files = sorted(
        name
        for name in tree
        if (
            (
                name.startswith(("src/", "tests/", "scripts/", "experiments/"))
                and name.endswith(".py")
            )
            or (name.startswith("tests/fixtures/") and name.endswith(".json"))
            or (name.startswith("configs/") and name.endswith(".yaml"))
            or name in ("uv.lock", "pyproject.toml")
        )
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update(
            subprocess.check_output(["git", "show", f"{args.benchmark_commit}:{name}"], cwd=ROOT)
        )
    for name in KERNEL_FILES:
        old = subprocess.check_output(["git", "show", f"{args.benchmark_commit}:{name}"], cwd=ROOT)
        if old != (ROOT / name).read_bytes():
            raise ValueError(f"kernel changed after benchmark: {name}")
    launch = {
        **vars(args),
        "benchmark_code_sha256": digest.hexdigest(),
        "benchmark_kernel_sha256": kernel_fingerprint(),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    run_id = "parallel-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    local = ROOT / "artifacts/replay-acceleration" / run_id
    local.mkdir(parents=True)
    record = {"run_id": run_id, **launch, "calls": {}}
    path = local / "launch.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    modal.Function.from_name(APP_NAME, "prepare").remote(run_id, launch)
    for mode in ("offline", "online"):
        call = modal.Function.from_name(APP_NAME, "replay").spawn(run_id, mode)
        record["calls"][mode] = call.object_id
        path.write_text(json.dumps(record, indent=2) + "\n")
    monitor = modal.Function.from_name(APP_NAME, "observe").spawn(run_id, record["calls"])
    record["monitor_call_id"] = monitor.object_id
    record["wandb_url"] = f"https://wandb.ai/cweill-self/janestreet-repro/runs/{run_id}"
    path.write_text(json.dumps(record, indent=2) + "\n")
    with output_volume.batch_upload() as upload:
        upload.put_file(path, f"/runs/{run_id}/calls.json")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
