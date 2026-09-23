"""Staged nine-epoch 17-seed training and paired later-date evaluation."""

import json
import re
from pathlib import Path

import modal

from scripts.modal_ensemble_run import image
from scripts.modal_online_sweep import tracker_for as base_tracker_for

app = modal.App("patrick-nine-epoch-ensemble")
output = modal.Volume.from_name("janestreet-patrick-nine-epoch-ensemble", create_if_missing=True)
research = modal.Volume.from_name("janestreet-patrick-reproduction")
baseline_volume = modal.Volume.from_name("janestreet-patrick-online-followup")
original_volume = modal.Volume.from_name("janestreet-patrick-ensemble")
tracking = modal.Volume.from_name("janestreet-wandb-monitor")
owners = modal.Dict.from_name("janestreet-nine-epoch-calls", create_if_missing=True)
volumes = {
    "/study": output,
    "/baseline": baseline_volume.with_mount_options(read_only=True),
    "/original": original_volume.with_mount_options(read_only=True),
    "/tracking": tracking,
    "/research": research.with_mount_options(read_only=True),
}
CONFIG = "configs/patrick_nine_epoch_ensemble.yaml"
REFERENCE_CONFIG = "configs/patrick_ensemble.yaml"
BASELINE = Path("/baseline/runs/ol-followup-20260920T070707Z")
ORIGINAL = Path("/original/runs/ensemble-20260919T030841Z")
DATASET_SHA = "915138f71873ba970eae0936a21a65b5ef0c19025a99392f89aa319cff2be79b"
CAPTURES = (5, 9)
secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


def tracker_for(run_id, suffix, config):
    tracker = base_tracker_for(run_id, suffix, config)
    tracker.name = f"Patrick nine epochs · 17 seeds · {suffix}"
    return tracker


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


def load_cache(root, launch, config, fold):
    from src.training.cache import prepare_cached

    class CacheOnly:
        def dates(self):
            return fold.train_dates

        def day(self, date):
            raise ValueError("verified preparation cache required before GPU training")

    prepared, record = prepare_cached(
        CacheOnly(),
        fold.train_dates,
        config.features,
        root / "preparation_cache",
        launch["dataset_sha256"],
    )
    if not record["hit"] or record["key"] != read(root / "preparation.json")["key"]:
        raise ValueError("GPU preparation cache mismatch")
    return prepared


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
    secrets=[secret],
    include_source=False,
)
def train_seed(run_id, seed):
    from time import perf_counter

    import torch

    from src.artifacts import write_json
    from src.epoch_study import snapshot_epoch
    from src.full_ensemble import check_epoch_five, require_seed_stage
    from src.monitoring import training_events
    from src.training.patrick import train_model
    from src.training.validation import PreparedValidation

    claim(run_id, f"seed_{seed}")
    root, launch, config, fold = load(run_id)
    if not read(root / "ready.json")["passed"]:
        raise ValueError("preflight required")
    pilot = read(root / "pilot_gate.json") if (root / "pilot_gate.json").exists() else None
    require_seed_stage(seed, pilot, launch["code_sha256"])
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
        prepared = load_cache(root, launch, config, fold)
        checkpoint = directory / "training.pt"
        reference = torch.load(
            BASELINE / "initial_checkpoint/weights.pt", map_location="cpu", weights_only=True
        )[seed]

        def progress(record):
            nonlocal logged
            if not record["checkpoint_written"]:
                return
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if check_epoch_five(saved, reference):
                write_json(directory / "epoch_five_control.json", {"passed": True, "seed": seed})
            snapshot_epoch(
                saved, prepared, config, seed, directory / "epochs", capture_epochs=CAPTURES
            )
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
            validation=PreparedValidation(root / "validation_cache"),
        )
        for epoch in CAPTURES:
            if not (directory / "epochs" / f"epoch_{epoch}" / "result.json").exists():
                raise ValueError("missing immutable epoch artifact")
        if not read(directory / "epoch_five_control.json")["passed"]:
            raise ValueError("epoch-five control required")
        result = {
            "status": "complete",
            "epoch_five_control_passed": True,
            "seed": seed,
            "epochs": config.training.epochs,
            "history": history,
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
    timeout=28800,
    max_containers=2,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def replay(run_id, stage, mode):
    import torch

    from src.artifacts import write_json
    from src.full_ensemble import replay_metrics
    from src.online_sweep import ReplayDayCache
    from src.parallel_replay import run_replay_mode

    claim(run_id, f"{stage}_{mode}")
    root, launch, config, fold = load(run_id)
    if stage not in ("pilot", "full") or mode not in ("offline", "online"):
        raise ValueError("invalid replay stage/mode")
    if not read(root / "ready.json")["passed"]:
        raise ValueError("verified preparation required")
    if stage == "full" and not read(root / "pilot_gate.json")["passed"]:
        raise ValueError("pilot gate required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dates = fold.replay_dates[:5] if stage == "pilot" else fold.replay_dates
    scored = dates if stage == "pilot" else fold.validation_dates
    directory = root / stage / mode
    tracker = tracker_for(run_id, f"{stage}-{mode}", {"stage": stage, "mode": mode, **launch})
    tracker.define_metric(f"{mode}/*", step_metric=f"{mode}/date_id")
    logged = int(tracker.summary.get(f"{mode}/date_id", dates[0] - 1))
    records = []

    def metric_record(date):
        record = read(directory / f"date_{date}" / "result.json")
        return {k: record[k] for k in ("date_id", "seconds", "diagnostic", "primary")}

    def emit():
        nonlocal logged
        date = records[-1]["date_id"]
        if date > logged:
            tracker.log(replay_metrics(records, mode), step=date)
            logged = date

    # Read an existing committed prefix once on resume. New days are appended in memory.
    for date in dates:
        marker = directory / "checkpoints" / f"date_{date}" / "replay.json"
        if not marker.exists():
            break
        if read(marker)["completed_date"] != date:
            raise ValueError("replay telemetry checkpoint mismatch")
        records.append(metric_record(date))
        emit()

    def progress(record):
        if "completed_date" in record:
            date = record["completed_date"]
            if date != dates[len(records)]:
                raise ValueError("replay telemetry date mismatch")
            records.append(metric_record(date))
            output.commit()
            emit()
        elif record.get("complete"):
            output.commit()
        print(json.dumps({"stage": stage, **record}), flush=True)

    try:
        result = run_replay_mode(
            ReplayDayCache(BASELINE / "replay_days"),
            directory,
            checkpoint=root / stage / "initial_checkpoint",
            mode=mode,
            replay_dates=dates,
            scored_dates=scored,
            device=config.training.device,
            provenance={
                "run_id": run_id,
                "stage": stage,
                "code_sha256": launch["code_sha256"],
                "dataset_sha256": launch["dataset_sha256"],
            },
            fast=True,
            progress=progress,
        )
        tracker.summary["final_r2"] = result["score"]
        tracker.finish()
        return {k: v for k, v in result.items() if k != "updates"}
    except Exception as error:
        write_json(
            root / stage / f"{mode}_failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        tracker.finish(exit_code=1)
        raise
    finally:
        output.commit()
        tracking.commit()


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=86400,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def coordinate(run_id, code_sha, commit):
    import tarfile
    import tempfile
    import time
    from dataclasses import asdict

    import wandb

    from src.artifacts import sha256_file, write_json
    from src.config import load_config
    from src.data.loader import ParquetSource, RestrictedDateSource
    from src.ensemble_run import assemble_ensemble, canonical, config_fingerprint
    from src.full_ensemble import compare_baseline, full_fold, verify_pair
    from src.online_sweep import ReplayDayCache
    from src.safety import safety_gate, source_fingerprint
    from src.training.cache import parquet_fingerprint, prepare_cached
    from src.training.validation import PreparedValidation, prepare_validation

    claim(run_id, "controller")
    if source_fingerprint() != code_sha:
        raise ValueError("local/deployed source mismatch")
    config, reference = load_config(CONFIG), load_config(REFERENCE_CONFIG)
    fold = full_fold(config, reference, range(1699))
    if canonical(reference.to_dict()) != read(ORIGINAL / "launch.json")["config"]:
        raise ValueError("historical five-epoch recipe differs")
    for path in (BASELINE, ORIGINAL):
        if read(path / "launch.json")["dataset_sha256"] != DATASET_SHA:
            raise ValueError("baseline dataset mismatch")
    launch = canonical(
        {
            "run_id": run_id,
            "code_sha256": code_sha,
            "source_commit": commit,
            "config": config.to_dict(),
            "config_sha256": config_fingerprint(config),
            "splits": asdict(fold),
            "dataset_sha256": DATASET_SHA,
            "baseline_run": "ol-followup-20260920T070707Z",
            "pilot_seeds": [0, 1, 2],
            "pilot_dates": list(fold.replay_dates[:5]),
            "max_training_gpus": 4,
            "max_replay_gpus": 2,
        }
    )
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

    def wait_jobs(phase, jobs):
        while True:
            output.reload()
            complete = [role for role, path in jobs.items() if path.exists()]
            state(phase, completed_jobs=complete, total_jobs=len(jobs))
            tracker.log({f"{phase}/completed_jobs": len(complete)})
            for role in jobs:
                # Surface remote errors even if a partial result file was written.
                try:
                    modal.FunctionCall.from_id(calls[role]).get(timeout=0)
                except TimeoutError:
                    if role in complete:
                        complete.remove(role)
            if len(complete) == len(jobs):
                break
            time.sleep(30)

    def assemble(stage, seeds):
        assemble_ensemble(
            [root / "seeds" / str(s) / "epochs/epoch_9" for s in seeds],
            config,
            root / stage / "initial_checkpoint",
            seeds=seeds,
        )
        output.commit()

    try:
        state("causal_gate")
        gate = safety_gate()
        if gate["code_and_tests_sha256"] != code_sha:
            raise ValueError("gate/source mismatch")
        write_json(root / "safety_gate.json", gate)
        project = Path(__file__).resolve().parents[1]
        if not (root / "source.tar.gz").exists():
            with tempfile.TemporaryDirectory(dir=root) as temp:
                staged = Path(temp) / "source.tar.gz"
                with tarfile.open(staged, "w:gz") as archive:
                    for folder in ("src", "scripts", "tests", "configs", "experiments"):
                        for path in sorted((project / folder).rglob("*")):
                            if path.is_file() and "__pycache__" not in path.parts:
                                archive.add(path, arcname=str(path.relative_to(project)))
                    for name in ("uv.lock", "pyproject.toml"):
                        archive.add(project / name, arcname=name)
                staged.rename(root / "source.tar.gz")
        output.commit()
        metadata = read(BASELINE / "initial_checkpoint/metadata.json")
        if (
            metadata["seeds"] != list(range(17))
            or metadata["online"] != canonical(config.to_dict()["online"])
            or sha256_file(BASELINE / "initial_checkpoint/weights.pt") != metadata["weights_sha256"]
        ):
            raise ValueError("baseline ensemble configuration/checksum mismatch")
        state("verifying_dataset")
        raw = Path("/research/datasets") / DATASET_SHA / "train.parquet"
        dataset_sha, files = parquet_fingerprint(raw)
        if dataset_sha != DATASET_SHA:
            raise ValueError("immutable raw dataset checksum mismatch")
        write_json(root / "dataset_verification.json", {"sha256": dataset_sha, "files": files})
        source = ParquetSource(raw)
        if full_fold(config, reference, source.dates()) != fold:
            raise ValueError("source date coverage mismatch")
        state("preparing_training_cache")
        prepared, preparation = prepare_cached(
            RestrictedDateSource(source, fold.train_dates),
            fold.train_dates,
            config.features,
            root / "preparation_cache",
            dataset_sha,
        )
        if prepared.features.state_dict() != metadata["feature_state"]:
            raise ValueError("new preprocessing differs from five-epoch control")
        write_json(root / "preparation.json", preparation)
        output.commit()
        state("preparing_validation_cache")
        if not (root / "validation_cache").exists():
            with tempfile.TemporaryDirectory(dir=root) as temp:
                val = prepare_validation(
                    RestrictedDateSource(source, fold.validation_dates),
                    fold.validation_dates,
                    prepared,
                    Path(temp) / "validation",
                )
                val.directory.rename(root / "validation_cache")
        PreparedValidation(root / "validation_cache").verify(prepared)
        output.commit()
        state("verifying_replay_cache")
        cache = ReplayDayCache(BASELINE / "replay_days")
        if cache.dates() != (1379, *fold.replay_dates):
            raise ValueError("replay cache coverage mismatch")
        for date in cache.dates():
            cache.day(date)
        write_json(
            root / "ready.json",
            {"passed": True, "dataset_sha256": dataset_sha, "preprocessing_matches_baseline": True},
        )
        output.commit()
        for seed in range(3):
            submit(f"seed_{seed}", train_seed, seed)
        wait_jobs(
            "pilot_training",
            {f"seed_{s}": root / "seeds" / str(s) / "result.json" for s in range(3)},
        )
        assemble("pilot", (0, 1, 2))
        for mode in ("offline", "online"):
            submit(f"pilot_{mode}", replay, "pilot", mode)
        wait_jobs(
            "pilot_replay",
            {f"pilot_{m}": root / "pilot" / m / "result.json" for m in ("offline", "online")},
        )
        output.reload()
        pilot = verify_pair(root / "pilot", fold.replay_dates[:5], fold.replay_dates[:5], (0, 1, 2))
        write_json(
            root / "pilot_gate.json",
            {**pilot, "code_sha256": code_sha, "score_used_for_selection": False},
        )
        output.commit()
        for seed in range(3, 17):
            submit(f"seed_{seed}", train_seed, seed)
        wait_jobs(
            "remaining_training",
            {f"seed_{s}": root / "seeds" / str(s) / "result.json" for s in range(3, 17)},
        )
        assemble("full", tuple(range(17)))
        for mode in ("offline", "online"):
            submit(f"full_{mode}", replay, "full", mode)
        wait_jobs(
            "full_replay",
            {f"full_{m}": root / "full" / m / "result.json" for m in ("offline", "online")},
        )
        output.reload()
        state("comparing")
        result = compare_baseline(
            root / "full",
            {"offline": BASELINE / "reference/offline", "online": BASELINE / "online"},
            fold,
            root / "comparison",
        )
        write_json(root / "result.json", result)
        tracker.log({"comparison/rolling_r2": wandb.Image(str(root / "comparison/comparison.png"))})
        for key in (
            "five_frozen",
            "five_online",
            "nine_frozen",
            "nine_online",
            "online_delta",
            "frozen_delta",
        ):
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
