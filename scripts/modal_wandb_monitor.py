"""Observe an existing Patrick run in W&B without modifying or restarting training.

Deploy this separate App, then invoke this module to spawn its CPU observer.
W&B receives scalar metrics, configuration/provenance, and the final diagnostic plot.
"""

import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "janestreet-wandb-monitor"
app = modal.App(APP_NAME)
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
tracking_volume = modal.Volume.from_name("janestreet-wandb-monitor", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extras=["dev"], frozen=True)
    .uv_pip_install("wandb==0.30.0")
    .env(
        {"PYTHONPATH": "/project:/project/scripts", "OMP_NUM_THREADS": "1", "WANDB_CONSOLE": "off"}
    )
    .workdir("/project")
    .add_local_file(ROOT / "src/monitoring.py", "/project/src/monitoring.py")
    .add_local_file(ROOT / "src/__init__.py", "/project/src/__init__.py")
    .add_local_file(
        ROOT / "scripts/modal_wandb_monitor.py", "/project/scripts/modal_wandb_monitor.py"
    )
    .add_local_file(ROOT / "pyproject.toml", "/project/pyproject.toml")
    .add_local_file(ROOT / "uv.lock", "/project/uv.lock")
)


@app.function(
    image=image,
    cpu=(0.25, 0.25),
    memory=(2048, 2048),
    timeout=43200,
    startup_timeout=900,
    min_containers=0,
    max_containers=1,
    retries=0,
    scaledown_window=2,
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    volumes={
        "/research": source_volume.with_mount_options(read_only=True),
        "/tracking": tracking_volume,
    },
    include_source=False,
)
def observe(run_id: str, source_call_id: str, entity: str, project: str, poll_seconds: int = 60):
    import time
    from datetime import UTC, datetime

    import torch
    import wandb

    from src.monitoring import evaluation_events, training_events

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) or poll_seconds < 15:
        raise ValueError("invalid run ID or polling interval")
    torch.set_num_threads(1)
    source_volume.reload()
    root = Path("/research/runs") / run_id
    identity = json.loads((root / "identity.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    config = identity["config"]
    days, epochs = len(splits["train_dates"]), config["training"]["epochs"]
    dates = tuple(splits["warmup_dates"] + splits["validation_dates"])
    total = days * epochs
    directory = Path("/tracking") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    tracking = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=f"Patrick OL · {run_id}",
        resume="allow",
        job_type="read-only-monitor",
        dir=str(directory),
        save_code=False,
        config={
            **config,
            "source_run_id": run_id,
            "source_call_id": source_call_id,
            "source_code_sha256": identity["code_sha256"],
            "dataset_sha256": identity["dataset_sha256"],
            "monitor_poll_seconds": poll_seconds,
            "rolling_window": 20,
        },
        settings=wandb.Settings(
            x_disable_stats=True,
            disable_git=True,
            disable_code=True,
            console="off",
            init_timeout=60,
        ),
    )
    tracking.define_metric("train/optimization_loss", step_metric="train/batch")
    tracking.define_metric("train/progress_fraction", step_metric="train/batch")
    tracking.define_metric("train/epoch_fraction", step_metric="train/batch")
    tracking.define_metric("train/epoch_mean_optimization_loss", step_metric="train/epoch")
    tracking.define_metric("offline/*", step_metric="eval/date_id")
    tracking.define_metric("online/*", step_metric="eval/date_id")
    tracking.summary["loss_note"] = (
        "Detached balancing normalizes optimization loss; this is not an error or learning curve. "
        "Use replay weighted zero-mean R2 to assess predictions. No evaluation during offline epochs."
    )
    tracking.summary["hardware_note"] = (
        "Observer CPU statistics disabled; this is not the trainer GPU."
    )
    tracking.summary["monitor_status"] = "running"
    (directory / "connection.json").write_text(json.dumps({"url": tracking.url, "run_id": run_id}))
    tracking_volume.commit()
    print(json.dumps({"wandb_url": tracking.url, "source_run_id": run_id}), flush=True)
    source_call = modal.FunctionCall.from_id(source_call_id)
    last_checkpoint = None
    started = time.monotonic()
    try:
        while time.monotonic() - started < 42000:
            # No open source file handles persist across reload; source mount is read-only.
            source_volume.reload()
            status = json.loads((root / "status.json").read_text())
            tracking.summary["source_phase"] = status["phase"]
            tracking.summary["observed_at_utc"] = datetime.now(UTC).isoformat()
            events = []
            checkpoint = root / "training.pt"
            if checkpoint.exists():
                stamp = (checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
                if stamp != last_checkpoint:
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    events.extend(training_events(saved, days_per_epoch=days, epochs=epochs))
                    tracking.summary["completed_batches"] = saved["completed"]
                    tracking.summary["total_batches"] = total
                    tracking.summary["epochs_completed"] = saved["epoch"]
                    last_checkpoint = stamp
                    del saved
            for mode in ("offline", "online"):
                events.extend(
                    evaluation_events(root, mode, replay_dates=dates, total_batches=total)
                )
            for event in sorted(events, key=lambda item: item["step"]):
                # W&B restores its next step when this observer is resumed. Custom axes
                # retain true batch/date positions; repeated snapshots aren't re-logged.
                if event["step"] >= tracking.step:
                    tracking.log(event["metrics"], step=event["step"])
            try:
                source_result = source_call.get(timeout=0)
            except TimeoutError:
                source_result = None
            if source_result is not None:
                # Reload once more on the next pass if the source finished after our snapshot.
                if not (root / "result.json").exists():
                    continue
                result = json.loads((root / "result.json").read_text())
                for mode in ("offline", "online"):
                    tracking.summary[f"{mode}/final_scored_r2"] = result[mode]["score"]
                terminal_step = 2 * total + 2 + 2 * len(dates)
                if terminal_step >= tracking.step:
                    tracking.log(
                        {"plots/online_learning": wandb.Image(str(root / "online_learning.png"))},
                        step=terminal_step,
                    )
                tracking.summary["monitor_status"] = "complete"
                tracking.finish()
                final = {"status": "complete", "url": tracking.url}
                (directory / "result.json").write_text(json.dumps(final))
                return final
            heartbeat = {
                "url": tracking.url,
                "source_status": status,
                "next_wandb_step": tracking.step,
                "observed_at_utc": datetime.now(UTC).isoformat(),
            }
            (directory / "status.json").write_text(json.dumps(heartbeat))
            tracking_volume.commit()
            time.sleep(poll_seconds)
        tracking.summary["monitor_status"] = "observer_timeout; source unaffected"
        tracking.finish()
        return {"status": "observer_timeout", "url": tracking.url}
    except Exception as error:
        tracking.summary["monitor_status"] = "observer_error; check source independently"
        tracking.summary["observer_error_type"] = type(error).__name__
        tracking.finish(exit_code=1)
        raise
    finally:
        tracking_volume.commit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-call-id", required=True)
    parser.add_argument("--entity", default="cweill-self")
    parser.add_argument("--project", default="janestreet-repro")
    args = parser.parse_args()
    call = modal.Function.from_name(APP_NAME, "observe").spawn(
        args.run_id,
        args.source_call_id,
        args.entity,
        args.project,
    )
    record = {
        **vars(args),
        "monitor_call_id": call.object_id,
        "wandb_url": f"https://wandb.ai/{args.entity}/{args.project}/runs/{args.run_id}",
    }
    path = ROOT / "artifacts/patrick-reproduction" / args.run_id / "wandb-monitor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
