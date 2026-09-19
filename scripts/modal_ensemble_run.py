"""Persistent four-L4 training pool, seed-0 reuse, and matched 17-member replay."""

import json
import re
from pathlib import Path

import modal

from scripts.modal_pilot import ROOT

APP_NAME = "patrick-seed-ensemble"
app = modal.App(APP_NAME)
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
output_volume = modal.Volume.from_name("janestreet-patrick-ensemble", create_if_missing=True)
tracking_volume = modal.Volume.from_name("janestreet-wandb-monitor")
owners = modal.Dict.from_name("janestreet-ensemble-calls", create_if_missing=True)
MOUNT = Path("/ensemble")
CONFIG = "configs/patrick_ensemble.yaml"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extras=["dev"], frozen=True)
    .uv_pip_install("wandb==0.30.0")
    .env(
        {
            "PYTHONPATH": "/project:/project/scripts",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "4",
            "POLARS_MAX_THREADS": "4",
            "WANDB_CONSOLE": "off",
        }
    )
    .workdir("/project")
)
for directory in ("src", "tests", "scripts", "experiments", "configs"):
    image = image.add_local_dir(
        ROOT / directory, f"/project/{directory}", ignore=["**/__pycache__/**"]
    )
for name in ("pyproject.toml", "uv.lock"):
    image = image.add_local_file(ROOT / name, f"/project/{name}")
volumes = {
    "/research": source_volume.with_mount_options(read_only=True),
    str(MOUNT): output_volume,
    "/tracking": tracking_volume,
}
secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


def claim(run_id, role):
    key, call_id = f"{run_id}/owner/{role}", modal.current_function_call_id()
    if not owners.put(key, call_id, skip_if_exists=True) and owners[key] != call_id:
        raise RuntimeError(f"another independent call owns {role}")


def load_launch(run_id):
    from src.config import load_config
    from src.ensemble_run import config_fingerprint
    from src.safety import source_fingerprint

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    output_volume.reload()
    root = MOUNT / "runs" / run_id
    launch = json.loads((root / "launch.json").read_text())
    config = load_config(CONFIG)
    if (
        config_fingerprint(config) != launch["config_sha256"]
        or source_fingerprint() != launch["code_sha256"]
    ):
        raise ValueError("deployment/config differs from verified launch")
    return root, launch, config


def load_cache(launch, config):
    from src.ensemble_run import load_verified_cache

    return load_verified_cache(
        Path(launch["cache_directory"]),
        launch["source_archive"],
        config,
        tuple(launch["splits"]["train_dates"]),
        launch["dataset_sha256"],
    )


def tracker_for(run_id, suffix, config, *, job_type):
    import wandb

    directory = Path("/tracking") / run_id / suffix
    directory.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        entity="cweill-self",
        project="janestreet-repro",
        id=f"{run_id}-{suffix}",
        name=f"Patrick 17 seeds · {suffix}",
        group=run_id,
        job_type=job_type,
        resume="allow",
        dir=str(directory),
        config=config,
        save_code=False,
        settings=wandb.Settings(
            x_disable_stats=True, disable_git=True, disable_code=True, console="off"
        ),
    )


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes=volumes,
    include_source=False,
)
def prepare(run_id: str, source_run: str, expected_code: str, commit: str):
    import shutil
    import tarfile
    from dataclasses import asdict

    import torch

    from scripts.modal_patrick_reproduction import verified_source
    from src.artifacts import load_predictor, sha256_file, write_json
    from src.config import load_config
    from src.cv import configured_folds
    from src.ensemble_run import canonical, config_fingerprint, load_verified_cache
    from src.safety import safety_gate

    if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", x) for x in (run_id, source_run)):
        raise ValueError("invalid run IDs")
    claim(run_id, "prepare")
    gate = safety_gate()
    if gate["code_and_tests_sha256"] != expected_code:
        raise ValueError("local and remote source differ")
    config = load_config(CONFIG)
    origin = Path("/research/runs") / source_run
    old = json.loads((origin / "identity.json").read_text())
    for key in ("features", "model", "training", "online", "cv"):
        if old["config"][key] != canonical(config.to_dict()[key]):
            raise ValueError(f"reused seed configuration differs: {key}")
    source = verified_source(old["dataset_sha256"])
    (fold,) = configured_folds(source.dates(), config.cv)
    cache_record = json.loads((origin / "preparation_cache.json").read_text())
    cache_directory = Path("/research/preparation_cache") / cache_record["key"]
    archive = Path("/research/launches") / source_run / "source.tar.gz"
    print("Verifying reusable cache contents and archived preparation source", flush=True)
    prepared = load_verified_cache(
        cache_directory, archive, config, fold.train_dates, old["dataset_sha256"]
    )
    initial = load_predictor(origin / "initial_checkpoint")
    saved = torch.load(origin / "training.pt", map_location="cpu", weights_only=True)
    if (
        initial.seeds != (0,)
        or saved["epoch"] != config.training.epochs
        or saved["position"] != 0
        or saved["completed"] != len(fold.train_dates) * config.training.epochs
        or initial.features.state_dict() != prepared.features.state_dict()
    ):
        raise ValueError("seed zero is not the matching completed offline model")
    for key, value in initial.models[0].state_dict().items():
        torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
    root = MOUNT / "runs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    launch = {
        "run_id": run_id,
        "source_run": source_run,
        "source_commit": commit,
        "dataset_sha256": old["dataset_sha256"],
        "cache_directory": str(cache_directory),
        "source_archive": str(archive),
        "config": config.to_dict(),
        "config_sha256": config_fingerprint(config),
        "code_sha256": expected_code,
        "splits": asdict(fold),
        "seeds": list(range(17)),
        "max_training_gpus": 4,
        "source_seed_zero_weights_sha256": sha256_file(origin / "initial_checkpoint/weights.pt"),
    }
    write_json(root / "launch.json", launch)
    write_json(root / "safety_gate.json", gate)
    for seed in launch["seeds"]:
        (root / "seeds" / str(seed)).mkdir(parents=True)
    shutil.copytree(origin / "initial_checkpoint", root / "seeds/0/checkpoint")
    write_json(
        root / "seeds/0/result.json",
        {
            "seed": 0,
            "status": "complete",
            "reused": True,
            "epochs": config.training.epochs,
            "config_sha256": config_fingerprint(config),
            "source_run": source_run,
            "source_weights_sha256": launch["source_seed_zero_weights_sha256"],
        },
    )
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive_out:
        for name in (
            "src",
            "tests",
            "scripts",
            "experiments",
            "configs",
            "uv.lock",
            "pyproject.toml",
        ):
            archive_out.add(
                Path("/project") / name,
                arcname=name,
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
    write_json(root / "status.json", {"phase": "prepared", "completed_seeds": [0]})
    output_volume.commit()
    return {
        "status": "prepared",
        "run_id": run_id,
        "cache_reused": True,
        "source_seed_zero_verified": True,
        "code_sha256": expected_code,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=1200,
    max_containers=1,
    retries=0,
    volumes=volumes,
    include_source=False,
)
def rehearse(run_id: str):
    import tempfile
    from dataclasses import replace

    import torch

    from src.artifacts import write_json
    from src.training.patrick import PreparedPatrick, train_model

    root, launch, config = load_launch(run_id)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    prepared = load_cache(launch, config)
    sample = PreparedPatrick(prepared.directory, prepared.dates[-3:], prepared.features)
    training = replace(config.training, epochs=2)
    expected, expected_history = train_model(sample, config.model, training=training, seed=1)

    class Interrupted(Exception):
        pass

    def interrupt(record):
        if record["completed_batches"] == 2:
            raise Interrupted()

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "training.pt"
        try:
            train_model(
                sample,
                config.model,
                training=training,
                seed=1,
                checkpoint_path=path,
                checkpoint_every=1,
                progress=interrupt,
            )
        except Interrupted:
            pass
        else:
            raise AssertionError("rehearsal did not interrupt")
        actual, history = train_model(
            sample,
            config.model,
            training=training,
            seed=1,
            checkpoint_path=path,
            checkpoint_every=1,
        )
    assert history == expected_history
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    record = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "dates": list(sample.dates),
        "epochs": 2,
        "resume_weights_history_exact": True,
        "code_sha256": launch["code_sha256"],
    }
    write_json(root / "rehearsal.json", record)
    output_volume.commit()
    return record


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=7200,
    max_containers=4,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def train_seed(run_id: str, seed: int):
    import tempfile
    from time import perf_counter

    import torch

    from src.artifacts import save_predictor, write_json
    from src.ensemble_run import config_fingerprint
    from src.monitoring import training_events
    from src.training.patrick import train_model

    claim(run_id, f"seed_{seed}")
    root, launch, config = load_launch(run_id)
    if seed not in launch["seeds"] or seed == 0:
        raise ValueError("only new configured seeds may be trained")
    if json.loads((root / "rehearsal.json").read_text())["status"] != "passed":
        raise ValueError("GPU recovery rehearsal required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    directory = root / "seeds" / str(seed)
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    tracker = tracker_for(
        run_id,
        f"seed-{seed:02d}",
        {**launch["config"], "seed": seed, "source_commit": launch["source_commit"]},
        job_type="ensemble-seed-training",
    )
    tracker.define_metric("train/*", step_metric="train/batch")
    started = perf_counter()
    try:
        write_json(
            directory / "status.json",
            {
                "phase": "verifying_cache",
                "seed": seed,
                "call_id": modal.current_function_call_id(),
                "wandb_url": tracker.url,
            },
        )
        output_volume.commit()
        prepared = load_cache(launch, config)
        checkpoint = directory / "training.pt"
        logged = int(tracker.summary.get("last_checkpoint_step", -1))

        def progress(record):
            nonlocal logged
            if not record["checkpoint_written"]:
                return
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            events = training_events(
                saved, days_per_epoch=len(prepared.dates), epochs=config.training.epochs
            )
            for event in events:
                if event["step"] > logged:
                    metrics = event["metrics"]
                    if "train/batch" not in metrics:
                        metrics["train/batch"] = int(metrics["train/epoch"]) * len(prepared.dates)
                    tracker.log(metrics, step=event["step"])
                    logged = event["step"]
            tracker.summary["last_checkpoint_step"] = logged
            state = {
                "phase": "training",
                "seed": seed,
                **record,
                "wall_seconds": perf_counter() - started,
                "wandb_url": tracker.url,
                "call_id": modal.current_function_call_id(),
            }
            write_json(directory / "status.json", state)
            output_volume.commit()
            print(json.dumps(state), flush=True)

        model, history = train_model(
            prepared,
            config.model,
            training=config.training,
            seed=seed,
            checkpoint_path=checkpoint,
            checkpoint_every=25,
            progress=progress,
        )
        if not (directory / "checkpoint").exists():
            with tempfile.TemporaryDirectory(prefix=".final-", dir=directory) as temp:
                staged = Path(temp) / "checkpoint"
                save_predictor(
                    staged, [model], prepared.features, prepared.scaler, config.online, [seed]
                )
                staged.rename(directory / "checkpoint")
        result = {
            "seed": seed,
            "status": "complete",
            "epochs": config.training.epochs,
            "config_sha256": config_fingerprint(config),
            "seconds": perf_counter() - started,
            "wandb_url": tracker.url,
            "history": history,
        }
        write_json(directory / "result.json", result)
        write_json(directory / "status.json", result)
        tracker.summary["status"] = "complete"
        tracker.finish()
        return result
    except Exception as error:
        write_json(
            directory / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        tracker.finish(exit_code=1)
        raise
    finally:
        output_volume.commit()
        tracking_volume.commit()


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
    include_source=False,
)
def replay(run_id: str, mode: str):
    import torch

    from scripts.modal_patrick_reproduction import verified_source
    from src.parallel_replay import run_replay_mode

    claim(run_id, f"replay_{mode}")
    root, launch, config = load_launch(run_id)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source = verified_source(launch["dataset_sha256"])

    def progress(record):
        if "completed_date" in record or record.get("complete"):
            output_volume.commit()
            print(json.dumps(record), flush=True)

    try:
        return run_replay_mode(
            source,
            root / mode,
            checkpoint=root / "initial_checkpoint",
            mode=mode,
            replay_dates=launch["splits"]["warmup_dates"] + launch["splits"]["validation_dates"],
            scored_dates=launch["splits"]["validation_dates"],
            device=config.training.device,
            provenance={
                "ensemble_run": run_id,
                "source_commit": launch["source_commit"],
                "code_sha256": launch["code_sha256"],
            },
            fast=True,
            progress=progress,
        )
    finally:
        output_volume.commit()


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(4096, 4096),
    timeout=64800,
    max_containers=1,
    min_containers=0,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def coordinate(run_id: str):
    import time

    import wandb

    from src.artifacts import write_json
    from src.ensemble_run import assemble_ensemble
    from src.monitoring import evaluation_events
    from src.parallel_replay import finish_comparison

    claim(run_id, "controller")
    root, launch, config = load_launch(run_id)
    rehearsal = json.loads((root / "rehearsal.json").read_text())
    if rehearsal["status"] != "passed" or rehearsal["code_sha256"] != launch["code_sha256"]:
        raise ValueError("matching successful GPU rehearsal required")
    tracker = tracker_for(run_id, "overview", launch, job_type="ensemble-training-and-replay")
    tracker.summary["reused_seed_zero"] = launch["source_run"]
    tracker.summary["max_concurrent_training_gpus"] = 4
    calls = {}

    def submit(role, function, *args):
        key = f"{run_id}/call/{role}"
        call_id = owners.get(key) or owners.get(f"{run_id}/owner/{role}")
        if not call_id:
            call_id = function.spawn(run_id, *args).object_id
            owners[key] = call_id
        calls[role] = call_id
        write_json(root / "calls.json", calls)
        output_volume.commit()
        return call_id

    try:
        for seed in launch["seeds"]:
            if seed:
                submit(f"seed_{seed}", train_seed, seed)
        while True:
            output_volume.reload()
            complete, failed, running = [0], {}, []
            for seed in launch["seeds"][1:]:
                directory = root / "seeds" / str(seed)
                if (directory / "result.json").exists():
                    complete.append(seed)
                    continue
                role = f"seed_{seed}"
                actual_owner = owners.get(f"{run_id}/owner/{role}")
                if actual_owner and actual_owner != calls[role]:
                    calls[role] = actual_owner
                    owners[f"{run_id}/call/{role}"] = actual_owner
                try:
                    modal.FunctionCall.from_id(calls[role]).get(timeout=0)
                except TimeoutError:
                    running.append(seed)
                except Exception as error:  # noqa: BLE001 - collect child failures while other seeds finish.
                    failed[str(seed)] = str(error)
            state = {
                "phase": "training",
                "completed_seeds": complete,
                "unfinished_seeds": running,
                "failed_seeds": failed,
                "wandb_url": tracker.url,
            }
            write_json(root / "status.json", state)
            output_volume.commit()
            tracker.log(
                {"ensemble/completed_seeds": len(complete), "ensemble/failed_seeds": len(failed)}
            )
            if len(complete) == len(launch["seeds"]):
                break
            if failed and not running:
                raise RuntimeError(f"seed training failed: {failed}")
            time.sleep(30)
        assemble_ensemble(
            [root / "seeds" / str(s) for s in launch["seeds"]], config, root / "initial_checkpoint"
        )
        output_volume.commit()
        replay_calls = {
            mode: submit(f"replay_{mode}", replay, mode) for mode in ("offline", "online")
        }
        for mode in replay_calls:
            tracker.define_metric(f"{mode}/*", step_metric=f"{mode}/date_id")
        seen = {
            mode: int(tracker.summary.get(f"{mode}/days_completed", 0)) for mode in replay_calls
        }
        dates = launch["splits"]["warmup_dates"] + launch["splits"]["validation_dates"]
        while True:
            output_volume.reload()
            finished = True
            for mode, call_id in replay_calls.items():
                for event in evaluation_events(root, mode, replay_dates=dates, total_batches=0):
                    metrics = event["metrics"]
                    if metrics[f"{mode}/days_completed"] > seen[mode]:
                        metrics[f"{mode}/date_id"] = metrics.pop("eval/date_id")
                        tracker.log(metrics)
                        seen[mode] = metrics[f"{mode}/days_completed"]
                try:
                    modal.FunctionCall.from_id(call_id).get(timeout=0)
                except TimeoutError:
                    finished = False
            write_json(
                root / "status.json",
                {
                    "phase": "replay",
                    "completed_seeds": launch["seeds"],
                    "replay_days_completed": seen,
                    "wandb_url": tracker.url,
                },
            )
            output_volume.commit()
            if finished:
                # Reload once more after the last remote result so final committed
                # days cannot race the observer's previous volume snapshot.
                output_volume.reload()
                for mode in replay_calls:
                    for event in evaluation_events(root, mode, replay_dates=dates, total_batches=0):
                        metrics = event["metrics"]
                        if metrics[f"{mode}/days_completed"] > seen[mode]:
                            metrics[f"{mode}/date_id"] = metrics.pop("eval/date_id")
                            tracker.log(metrics)
                            seen[mode] = metrics[f"{mode}/days_completed"]
                result = finish_comparison(
                    root, scored_start=launch["splits"]["validation_dates"][0]
                )
                for mode in replay_calls:
                    tracker.summary[f"{mode}/final_scored_r2"] = result[mode]["score"]
                tracker.log(
                    {"plots/online_learning": wandb.Image(str(root / "online_learning.png"))}
                )
                write_json(root / "status.json", {"phase": "complete", "wandb_url": tracker.url})
                tracker.summary["status"] = "complete"
                tracker.finish()
                return {"status": "complete", "wandb_url": tracker.url}
            time.sleep(30)
    except Exception as error:
        write_json(
            root / "controller_failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        tracker.finish(exit_code=1)
        raise
    finally:
        output_volume.commit()
        tracking_volume.commit()
