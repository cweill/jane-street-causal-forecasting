"""One L4 replay of the completed 17-member ensemble at online LR 1e-4."""

import json
import re
from pathlib import Path

import modal

from scripts.modal_ensemble_run import image

app = modal.App("patrick-online-followup")
output = modal.Volume.from_name("janestreet-patrick-online-followup", create_if_missing=True)
ensemble = modal.Volume.from_name("janestreet-patrick-ensemble")
source = modal.Volume.from_name("janestreet-patrick-reproduction")
tracking = modal.Volume.from_name("janestreet-wandb-monitor")
owners = modal.Dict.from_name("janestreet-followup-calls", create_if_missing=True)
volumes = {
    "/followup": output,
    "/ensemble": ensemble.with_mount_options(read_only=True),
    "/research": source.with_mount_options(read_only=True),
    "/tracking": tracking,
}
ORIGINAL = "ensemble-20260919T030841Z"
DATASET = "915138f71873ba970eae0936a21a65b5ef0c19025a99392f89aa319cff2be79b"
secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


def claim(run_id, role):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    key, call = f"{run_id}/owner/{role}", modal.current_function_call_id()
    if not owners.put(key, call, skip_if_exists=True) and owners[key] != call:
        raise ValueError("another call owns this role")


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=28800,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    include_source=False,
)
def replay(run_id):
    import torch

    from src.artifacts import write_json
    from src.online_followup import check_first_day, read
    from src.online_sweep import ReplayDayCache
    from src.parallel_replay import run_replay_mode
    from src.safety import source_fingerprint

    claim(run_id, "replay")
    output.reload()
    root = Path("/followup/runs") / run_id
    launch, record = read(root / "launch.json"), read(root / "followup.json")
    if source_fingerprint() != launch["code_sha256"] or not read(root / "ready.json")["passed"]:
        raise ValueError("prepared source identity required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    first = root / "online" / f"date_{record['replay_dates'][0]}.parquet"
    if first.exists():
        check_first_day(root)

    def progress(value):
        if value.get("completed_date") == record["replay_dates"][0]:
            check_first_day(root)
            write_json(
                root / "first_day_parity.json",
                {"passed": True, "date_id": record["replay_dates"][0]},
            )
        if "completed_date" in value or value.get("complete"):
            output.commit()
            print(json.dumps(value), flush=True)

    try:
        return run_replay_mode(
            ReplayDayCache(root / "replay_days"),
            root / "online",
            checkpoint=root / "initial_checkpoint",
            mode="online",
            replay_dates=record["replay_dates"],
            scored_dates=record["scored_dates"],
            device="cuda",
            provenance=launch,
            fast=True,
            progress=progress,
        )
    except Exception as error:
        write_json(
            root / "replay_failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        output.commit()


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=43200,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def coordinate(run_id, code_sha, commit):
    import time

    import wandb

    from scripts.modal_patrick_reproduction import verified_source
    from src.artifacts import write_json
    from src.monitoring import evaluation_events
    from src.online_followup import finish_followup, prepare_followup, read
    from src.online_sweep import prepare_replay_days
    from src.safety import safety_gate, source_fingerprint

    claim(run_id, "controller")
    if source_fingerprint() != code_sha:
        raise ValueError("deployed source mismatch")
    root = Path("/followup/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    origin = Path("/ensemble/runs") / ORIGINAL
    launch = {
        "run_id": run_id,
        "source_commit": commit,
        "code_sha256": code_sha,
        "original_run": ORIGINAL,
        "dataset_sha256": DATASET,
        "learning_rate": 1e-4,
        "reset_daily_optimizer": False,
        "max_gpus": 1,
    }
    if (root / "launch.json").exists() and read(root / "launch.json") != launch:
        raise ValueError("launch identity mismatch")
    write_json(root / "launch.json", launch)
    output.commit()
    directory = Path("/tracking") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    tracker = wandb.init(
        entity="cweill-self",
        project="janestreet-repro",
        id=run_id,
        group=run_id,
        name="Patrick 17 seeds · online LR 1e-4 follow-up",
        job_type="online-followup",
        resume="allow",
        dir=str(directory),
        config=launch,
        save_code=False,
        settings=wandb.Settings(
            x_disable_stats=True, disable_git=True, disable_code=True, console="off"
        ),
    )

    def state(phase, **values):
        tracker.summary["phase"] = phase
        write_json(root / "status.json", {"phase": phase, "wandb_url": tracker.url, **values})
        output.commit()

    try:
        state("causal_gate")
        gate = safety_gate()
        if gate["code_and_tests_sha256"] != code_sha:
            raise ValueError("gate fingerprint mismatch")
        write_json(root / "safety_gate.json", gate)
        output.commit()
        record = prepare_followup(origin, root)
        metadata = read(root / "initial_checkpoint/metadata.json")
        if (
            record["replay_dates"] != list(range(1380, 1699))
            or record["scored_dates"] != list(range(1500, 1699))
            or metadata["seeds"] != list(range(17))
        ):
            raise ValueError("expected original 17-seed evaluation protocol")
        if read(origin / "launch.json")["dataset_sha256"] != DATASET:
            raise ValueError("original dataset identity differs")
        state("preparing_replay_cache")
        verified = verified_source(DATASET)
        prepare_replay_days(verified, (1379, *record["replay_dates"]), root / "replay_days")
        write_json(
            root / "ready.json",
            {"passed": True, "code_sha256": code_sha, "dataset_sha256": DATASET},
        )
        output.commit()
        for label, mode in [("frozen", "offline"), ("online_5e-4", "online")]:
            tracker.define_metric(f"{label}/*", step_metric=f"{label}/date_id")
            seen = int(tracker.summary.get(f"{label}/days_completed", 0))
            for event in evaluation_events(
                origin, mode, replay_dates=record["replay_dates"], total_batches=0
            ):
                m = event["metrics"]
                if m[f"{mode}/days_completed"] > seen:
                    tracker.log(
                        {
                            f"{label}/date_id": m["eval/date_id"],
                            **{
                                f"{label}/{k.split('/', 1)[1]}": v
                                for k, v in m.items()
                                if k.startswith(mode + "/")
                            },
                        }
                    )
        key = f"{run_id}/call/replay"
        call = owners.get(key) or owners.get(f"{run_id}/owner/replay")
        if not call:
            call = replay.spawn(run_id).object_id
            owners[key] = call
        write_json(root / "calls.json", {"replay": call})
        output.commit()
        tracker.define_metric("online_1e-4/*", step_metric="online_1e-4/date_id")
        seen = int(tracker.summary.get("online_1e-4/days_completed", 0))
        while True:
            output.reload()
            for event in evaluation_events(
                root, "online", replay_dates=record["replay_dates"], total_batches=0
            ):
                m = event["metrics"]
                if m["online/days_completed"] > seen:
                    tracker.log(
                        {
                            "online_1e-4/date_id": m["eval/date_id"],
                            **{
                                f"online_1e-4/{k.split('/', 1)[1]}": v
                                for k, v in m.items()
                                if k.startswith("online/")
                            },
                        }
                    )
                    seen = m["online/days_completed"]
            state("replaying", completed_days=seen, total_days=len(record["replay_dates"]))
            try:
                modal.FunctionCall.from_id(call).get(timeout=0)
                if seen == len(record["replay_dates"]):
                    break
            except TimeoutError:
                pass
            time.sleep(30)
        output.reload()
        result = finish_followup(root)
        tracker.log({"comparison/rolling_r2": wandb.Image(str(root / "comparison.png"))})
        for label in ("frozen", "online_5e-4", "online_1e-4"):
            tracker.summary[f"{label}/final_r2"] = result[label]["score"]
        for key in ("delta_from_frozen", "delta_from_previous_online"):
            tracker.summary[key] = result[key]
        state("complete")
        tracker.finish()
        return {"status": "complete", "wandb_url": tracker.url}
    except Exception as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "message": str(error)})
        state("failed")
        tracker.finish(exit_code=1)
        raise
    finally:
        output.commit()
        tracking.commit()
