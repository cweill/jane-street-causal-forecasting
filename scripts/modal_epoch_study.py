"""Three four-epoch training jobs, four paired replays, and reused epoch-five results."""

import json
import re
from dataclasses import replace
from pathlib import Path

import modal

from scripts.modal_ensemble_run import image
from scripts.modal_online_refinement import verify_runtime_archive
from scripts.modal_online_sweep import tracker_for

app = modal.App("patrick-epoch-study")
output = modal.Volume.from_name("janestreet-patrick-epoch-study", create_if_missing=True)
original = modal.Volume.from_name("janestreet-patrick-ol-sweep")
tracking = modal.Volume.from_name("janestreet-wandb-monitor")
owners = modal.Dict.from_name("janestreet-epoch-study-calls", create_if_missing=True)
volumes = {
    "/study": output,
    "/original": original.with_mount_options(read_only=True),
    "/tracking": tracking,
}
ORIGINAL = "ol-sweep-20260919T194601Z"
ORIGIN = Path("/original/runs") / ORIGINAL
CONFIG = "configs/patrick_epoch_study.yaml"
secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


def read(path):
    return json.loads(Path(path).read_text())


def claim(run_id, role):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    key, call = f"{run_id}/owner/{role}", modal.current_function_call_id()
    if not owners.put(key, call, skip_if_exists=True) and owners[key] != call:
        raise ValueError("another call owns role")


def load(run_id):
    from src.config import load_config
    from src.cv import TemporalFold
    from src.ensemble_run import config_fingerprint
    from src.safety import source_fingerprint

    output.reload()
    root = Path("/study/runs") / run_id
    launch = read(root / "launch.json")
    config = load_config(CONFIG)
    if (
        source_fingerprint() != launch["code_sha256"]
        or config_fingerprint(config) != launch["config_sha256"]
    ):
        raise ValueError("deployment/configuration differs from launch")
    fold = TemporalFold(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in launch["splits"].items()}
    )
    return root, launch, config, fold


def load_cache(launch, config, fold):
    from src.ensemble_run import load_verified_cache

    return load_verified_cache(
        Path("/original/preparation_cache") / launch["cache_key"],
        ORIGIN / "source.tar.gz",
        config,
        fold.train_dates,
        launch["dataset_sha256"],
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=10800,
    max_containers=3,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def train_seed(run_id, seed):
    from time import perf_counter

    import torch

    from src.artifacts import write_json
    from src.epoch_study import check_history, snapshot_epoch
    from src.monitoring import training_events
    from src.training.patrick import train_model
    from src.training.validation import PreparedValidation

    claim(run_id, f"seed_{seed}")
    root, launch, config, fold = load(run_id)
    if seed not in (0, 1, 2) or not read(root / "ready.json")["passed"]:
        raise ValueError("preflight/seed mismatch")
    directory = root / "seeds" / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        return read(directory / "result.json")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tracker = tracker_for(run_id, f"seed-{seed:02d}", {**config.to_dict(), "seed": seed})
    tracker.define_metric("train/*", step_metric="train/batch")
    tracker.define_metric("val/*", step_metric="val/epoch")
    for name in ("mean_optimization_loss", "mean_unbalanced_loss", "responder_6_r2"):
        tracker.define_metric(f"train/epoch_{name}", step_metric="train/epoch")
    start = perf_counter()
    logged = int(tracker.summary.get("last_checkpoint_step", -1))
    try:
        write_json(
            directory / "status.json",
            {"phase": "verifying_cache", "seed": seed, "wandb_url": tracker.url},
        )
        output.commit()
        prepared = load_cache(launch, config, fold)
        checkpoint = directory / "training.pt"
        reference = read(ORIGIN / "seeds" / str(seed) / "result.json")["history"]

        def progress(record):
            nonlocal logged
            if not record["checkpoint_written"]:
                return
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            check_history(saved["history"], reference)
            snapshot_epoch(saved, prepared, config, seed, directory / "epochs")
            for event in training_events(
                saved, days_per_epoch=len(prepared.dates), epochs=config.training.epochs
            ):
                if event["step"] > logged:
                    tracker.log(event["metrics"], step=event["step"])
                    logged = event["step"]
            tracker.summary["last_checkpoint_step"] = logged
            state = {
                "phase": "training",
                **record,
                "seed": seed,
                "reference_epochs_verified": len(saved["history"]),
                "wall_seconds": perf_counter() - start,
                "wandb_url": tracker.url,
            }
            write_json(directory / "history.json", saved["history"])
            write_json(directory / "status.json", state)
            output.commit()
            print(json.dumps(state), flush=True)

        _, history = train_model(
            prepared,
            config.model,
            training=config.training,
            seed=seed,
            checkpoint_path=checkpoint,
            progress=progress,
            validation=PreparedValidation(ORIGIN / "validation_cache"),
        )
        for epoch in (3, 4):
            if not (directory / "epochs" / f"epoch_{epoch}" / "result.json").exists():
                raise ValueError("missing immutable epoch artifact")
        result = {
            "status": "complete",
            "seed": seed,
            "epochs": 4,
            "history": history,
            "reference_history_matched": True,
            "seconds": perf_counter() - start,
            "wandb_url": tracker.url,
        }
        write_json(directory / "result.json", result)
        tracker.finish()
        return result
    except Exception as error:
        write_json(
            directory / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        tracker.finish(exit_code=1)
        raise
    finally:
        output.commit()
        tracking.commit()


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=21600,
    max_containers=4,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    include_source=False,
)
def replay(run_id, epoch, mode):
    import torch

    from src.artifacts import write_json
    from src.online_sweep import ReplayDayCache
    from src.parallel_replay import run_replay_mode

    claim(run_id, f"e{epoch}_{mode}")
    root, launch, config, fold = load(run_id)
    if (
        epoch not in (3, 4)
        or mode not in ("offline", "online")
        or not read(root / "ready.json")["passed"]
    ):
        raise ValueError("preflight/epoch/mode mismatch")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    directory = root / "epochs" / f"epoch_{epoch}"

    def progress(record):
        if "completed_date" in record or record.get("complete"):
            output.commit()
            print(json.dumps({"epoch": epoch, **record}), flush=True)

    try:
        return run_replay_mode(
            ReplayDayCache(ORIGIN / "replay_days"),
            directory / mode,
            checkpoint=directory / "initial_checkpoint",
            mode=mode,
            replay_dates=fold.replay_dates,
            scored_dates=fold.validation_dates,
            device=config.training.device,
            provenance={
                "run_id": run_id,
                "code_sha256": launch["code_sha256"],
                "dataset_sha256": launch["dataset_sha256"],
                "offline_epochs": epoch,
            },
            fast=True,
            progress=progress,
        )
    except Exception as error:
        write_json(
            directory / f"{mode}_failure.json",
            {"type": type(error).__name__, "message": str(error)},
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

    from src.artifacts import write_json
    from src.config import load_config
    from src.cv import TemporalFold
    from src.ensemble_run import assemble_ensemble, canonical, config_fingerprint
    from src.epoch_study import compare_epochs
    from src.monitoring import evaluation_events
    from src.online_refinement import reused_trials
    from src.online_sweep import ReplayDayCache, summarize_trials
    from src.safety import safety_gate, source_fingerprint
    from src.training.validation import PreparedValidation

    claim(run_id, "controller")
    if source_fingerprint() != code_sha:
        raise ValueError("local/deployed source mismatch")
    config = load_config(CONFIG)
    origin = read(ORIGIN / "launch.json")
    expected = origin["config"]
    expected["name"] = config.name
    expected["training"]["epochs"] = 4
    expected["online"]["learning_rate"] = 1e-4
    if canonical(expected) != canonical(config.to_dict()):
        raise ValueError("only offline epoch budget and online LR may change")
    fold = TemporalFold(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in origin["splits"].items()}
    )
    launch = {
        "run_id": run_id,
        "code_sha256": code_sha,
        "source_commit": commit,
        "config": config.to_dict(),
        "config_sha256": config_fingerprint(config),
        "splits": origin["splits"],
        "original_run": ORIGINAL,
        "dataset_sha256": origin["dataset_sha256"],
        "cache_key": read(ORIGIN / "preparation.json")["key"],
        "captured_epochs": [3, 4],
        "reused_epoch": 5,
        "max_training_gpus": 3,
        "max_replay_gpus": 4,
    }
    launch = canonical(launch)
    root = Path("/study/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    if (root / "launch.json").exists() and read(root / "launch.json") != launch:
        raise ValueError("launch identity mismatch")
    write_json(root / "launch.json", launch)
    output.commit()
    tracker = tracker_for(run_id, "overview", launch)
    calls = {}

    def state(phase, **values):
        tracker.summary["phase"] = phase
        write_json(root / "status.json", {"phase": phase, "wandb_url": tracker.url, **values})
        output.commit()

    def submit(role, function, *args):
        key = f"{run_id}/call/{role}"
        call = owners.get(key) or owners.get(f"{run_id}/owner/{role}")
        if not call:
            call = function.spawn(run_id, *args).object_id
            owners[key] = call
        calls[role] = call
        write_json(root / "calls.json", calls)
        output.commit()
        return call

    try:
        state("causal_gate")
        gate = safety_gate()
        if gate["code_and_tests_sha256"] != code_sha:
            raise ValueError("gate/source mismatch")
        write_json(root / "safety_gate.json", gate)
        output.commit()
        parity = verify_runtime_archive()
        write_json(root / "runtime_parity.json", parity)
        if (
            fold.train_dates != tuple(range(1060))
            or fold.warmup_dates != tuple(range(1060, 1180))
            or fold.validation_dates != tuple(range(1180, 1380))
            or config.ensemble.seeds != (0, 1, 2)
        ):
            raise ValueError("fixed development protocol required")
        state("verifying_caches")
        prepared = load_cache(launch, config, fold)
        PreparedValidation(ORIGIN / "validation_cache").verify(prepared)
        cache = ReplayDayCache(ORIGIN / "replay_days")
        if cache.dates() != (1059, *fold.replay_dates):
            raise ValueError("replay cache coverage mismatch")
        for date in cache.dates():
            cache.day(date)
        trials = reused_trials()
        roots = {t.name: ORIGIN / "trials" / t.name for t in trials}
        old_prov = {
            "run_id": ORIGINAL,
            "code_sha256": origin["code_sha256"],
            "dataset_sha256": origin["dataset_sha256"],
        }
        summarize_trials(
            root / "reference_check",
            trials,
            fold,
            trial_directories=roots,
            expected_provenance={t.name: old_prov for t in trials},
        )
        write_json(root / "ready.json", {"passed": True, "runtime_parity": parity})
        output.commit()
        for seed in config.ensemble.seeds:
            submit(f"seed_{seed}", train_seed, seed)
        while True:
            output.reload()
            complete = [
                s
                for s in config.ensemble.seeds
                if (root / "seeds" / str(s) / "result.json").exists()
            ]
            state("training", completed_seeds=complete)
            tracker.log({"training/completed_seeds": len(complete)})
            if len(complete) == 3:
                break
            for seed in config.ensemble.seeds:
                try:
                    modal.FunctionCall.from_id(calls[f"seed_{seed}"]).get(timeout=0)
                except TimeoutError:
                    pass
            time.sleep(30)
        for epoch in (3, 4):
            epoch_config = replace(config, training=replace(config.training, epochs=epoch))
            assemble_ensemble(
                [
                    root / "seeds" / str(s) / "epochs" / f"epoch_{epoch}"
                    for s in config.ensemble.seeds
                ],
                epoch_config,
                root / "epochs" / f"epoch_{epoch}" / "initial_checkpoint",
            )
        output.commit()
        paths = {
            epoch: {
                mode: root / "epochs" / f"epoch_{epoch}" / mode for mode in ("offline", "online")
            }
            for epoch in (3, 4)
        }
        paths[5] = {
            "offline": ORIGIN / "trials/frozen/offline",
            "online": ORIGIN / "trials/lr_1e-4_persistent/online",
        }
        for epoch in (3, 4):
            for mode in ("offline", "online"):
                submit(f"e{epoch}_{mode}", replay, epoch, mode)
        seen = {}
        for epoch, pair in paths.items():
            for mode in pair:
                label = f"epoch_{epoch}_{'frozen' if mode == 'offline' else 'online'}"
                tracker.define_metric(f"{label}/*", step_metric=f"{label}/date_id")
                seen[label] = int(tracker.summary.get(f"{label}/days_completed", 0))
        while True:
            output.reload()
            for epoch, pair in paths.items():
                for mode, path in pair.items():
                    label = f"epoch_{epoch}_{'frozen' if mode == 'offline' else 'online'}"
                    for event in evaluation_events(
                        path.parent, mode, replay_dates=fold.replay_dates, total_batches=0
                    ):
                        m = event["metrics"]
                        if m[f"{mode}/days_completed"] > seen[label]:
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
                            seen[label] = m[f"{mode}/days_completed"]
            state("replaying", days_completed=seen)
            done = True
            for key, call in calls.items():
                if key.startswith("e"):
                    try:
                        modal.FunctionCall.from_id(call).get(timeout=0)
                    except TimeoutError:
                        done = False
            if done and all(v == len(fold.replay_dates) for v in seen.values()):
                break
            time.sleep(30)
        output.reload()
        result = compare_epochs(paths, fold, root / "comparison")
        tracker.log(
            {
                "comparison/rolling_r2": wandb.Image(
                    str(root / "comparison/rolling_comparison.png")
                ),
                "comparison/epoch_scores": wandb.Table(
                    columns=["epoch", "frozen_r2", "online_r2", "delta_r2"],
                    data=[
                        [r[k] for k in ("epoch", "frozen_r2", "online_r2", "delta_r2")]
                        for r in result["epochs"]
                    ],
                ),
            }
        )
        for r in result["epochs"]:
            for k in ("frozen_r2", "online_r2", "delta_r2"):
                tracker.summary[f"epoch_{r['epoch']}/{k}"] = r[k]
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
