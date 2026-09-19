"""Persistent, bounded development study of online LR and Adam reset policy."""

import json
import re
from dataclasses import asdict
from pathlib import Path

import modal

from scripts.modal_ensemble_run import image

app = modal.App("patrick-ol-sweep")
output_volume = modal.Volume.from_name("janestreet-patrick-ol-sweep", create_if_missing=True)
source_volume = modal.Volume.from_name("janestreet-patrick-reproduction")
tracking_volume = modal.Volume.from_name("janestreet-wandb-monitor")
owners = modal.Dict.from_name("janestreet-ol-sweep-calls", create_if_missing=True)
volumes = {
    "/research": source_volume.with_mount_options(read_only=True),
    "/sweep": output_volume,
    "/tracking": tracking_volume,
}
secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])
CONFIG = "configs/patrick_ol_sweep.yaml"


def claim(run_id, role):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    key, call_id = f"{run_id}/owner/{role}", modal.current_function_call_id()
    if not owners.put(key, call_id, skip_if_exists=True) and owners[key] != call_id:
        raise ValueError(f"another call owns {role}")


def load_launch(run_id):
    from src.config import load_config
    from src.ensemble_run import config_fingerprint
    from src.safety import source_fingerprint

    output_volume.reload()
    root = Path("/sweep/runs") / run_id
    launch = json.loads((root / "launch.json").read_text())
    config = load_config(CONFIG)
    if (
        source_fingerprint() != launch["code_sha256"]
        or config_fingerprint(config) != launch["config_sha256"]
    ):
        raise ValueError("deployment/config differs from verified launch")
    return root, launch, config


def prepared_for(root, launch, config):
    from src.data.loader import ParquetSource, RestrictedDateSource
    from src.training.cache import prepare_cached

    record = json.loads((root / "preparation.json").read_text())
    if not (Path(record["directory"]) / "manifest.json").is_file():
        raise ValueError("CPU preparation has not completed")
    source = ParquetSource(Path("/research/datasets") / launch["dataset_sha256"] / "train.parquet")
    dates = tuple(launch["splits"]["train_dates"])
    prepared, actual = prepare_cached(
        RestrictedDateSource(source, dates),
        dates,
        config.features,
        Path(record["directory"]).parent,
        launch["dataset_sha256"],
    )
    if not actual["hit"] or actual["key"] != record["key"]:
        raise ValueError("CPU/GPU cache identity mismatch")
    return prepared


def tracker_for(run_id, suffix, config):
    import wandb

    directory = Path("/tracking") / run_id / suffix
    directory.mkdir(parents=True, exist_ok=True)
    tracker = wandb.init(
        entity="cweill-self",
        project="janestreet-repro",
        id=f"{run_id}-{suffix}",
        group=run_id,
        name=f"Patrick OL sensitivity · {suffix}",
        job_type="online-sensitivity",
        resume="allow",
        dir=str(directory),
        config=config,
        save_code=False,
        settings=wandb.Settings(
            x_disable_stats=True, disable_git=True, disable_code=True, console="off"
        ),
    )
    return tracker


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=10800,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    include_source=False,
)
def prepare(run_id):
    import tarfile

    from scripts.modal_patrick_reproduction import verified_source
    from src.artifacts import write_json
    from src.cv import configured_folds
    from src.data.loader import RestrictedDateSource
    from src.online_sweep import prepare_replay_days
    from src.safety import safety_gate
    from src.training.cache import prepare_cached
    from src.training.validation import PreparedValidation, prepare_validation

    claim(run_id, "prepare")
    root, launch, config = load_launch(run_id)
    gate = safety_gate()
    if gate["code_and_tests_sha256"] != launch["code_sha256"]:
        raise ValueError("remote gate/source mismatch")
    write_json(root / "safety_gate.json", gate)
    output_volume.commit()
    source = verified_source(launch["dataset_sha256"])
    (fold,) = configured_folds(source.dates(), config.cv)
    if json.loads(json.dumps(asdict(fold))) != launch["splits"]:
        raise ValueError("actual source dates differ from declared split")
    print("Building development-only normalizer and daily training cache", flush=True)
    prepared, cache = prepare_cached(
        RestrictedDateSource(source, fold.train_dates),
        fold.train_dates,
        config.features,
        Path("/sweep/preparation_cache"),
        launch["dataset_sha256"],
    )
    write_json(root / "preparation.json", cache)
    output_volume.commit()
    print("Caching immutable replay days", flush=True)
    replay_source = prepare_replay_days(
        source, (fold.train_dates[-1], *fold.replay_dates), root / "replay_days"
    )
    if not (root / "validation_cache").exists():
        prepare_validation(
            replay_source, fold.validation_dates, prepared, root / "validation_cache"
        )
    PreparedValidation(root / "validation_cache").verify(prepared)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for name in (
            "src",
            "tests",
            "scripts",
            "experiments",
            "configs",
            "uv.lock",
            "pyproject.toml",
        ):
            archive.add(
                Path("/project") / name,
                arcname=name,
                filter=lambda info: None if "__pycache__" in info.name else info,
            )
    record = {
        "status": "prepared",
        "cache": cache,
        "training_dates": [fold.train_dates[0], fold.train_dates[-1]],
        "replay_dates": [fold.replay_dates[0], fold.replay_dates[-1]],
    }
    write_json(root / "ready.json", record)
    output_volume.commit()
    return record


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=1200,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    volumes=volumes,
    include_source=False,
)
def rehearse(run_id):
    import tempfile
    from dataclasses import replace

    import torch

    from src.artifacts import write_json
    from src.data.api_simulator import APISimulator
    from src.online_sweep import ReplayDayCache
    from src.training.patrick import PatrickPredictor, PreparedPatrick, train_model
    from src.training.validation import evaluate_validation, prepare_validation

    claim(run_id, "rehearse")
    root, launch, config = load_launch(run_id)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    prepared = prepared_for(root, launch, config)
    sample = PreparedPatrick(prepared.directory, prepared.dates[-3:], prepared.features)
    training = replace(config.training, epochs=2)
    expected, history = train_model(sample, config.model, training=training, seed=0)
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "training.pt"

        def stop(record):
            if record["completed_batches"] == 2:
                raise InterruptedError()

        try:
            train_model(
                sample,
                config.model,
                training=training,
                seed=0,
                checkpoint_path=checkpoint,
                checkpoint_every=1,
                progress=stop,
            )
        except InterruptedError:
            pass
        else:
            raise AssertionError("recovery rehearsal did not interrupt")
        actual, resumed = train_model(
            sample, config.model, training=training, seed=0, checkpoint_path=checkpoint
        )
        assert history == resumed
        for k, v in expected.state_dict().items():
            torch.testing.assert_close(v, actual.state_dict()[k], rtol=0, atol=0)
        source = ReplayDayCache(root / "replay_days")
        dates = tuple(launch["splits"]["validation_dates"][:2])
        validation = prepare_validation(source, dates, prepared, Path(temporary) / "validation")
        measured = evaluate_validation(actual, validation)
        predictor = PatrickPredictor(
            [actual], prepared.features, replace(config.online, enabled=False), [0]
        )
        predictor.fast_inference = True
        reference = (
            APISimulator(source, dates, prepartition_truth=True).run(predictor.predict).metric
        )
        assert abs(measured["r2"] - reference.score) < 1e-6
    record = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "resume_weights_history_exact": True,
        "validation_r2": measured["r2"],
        "streamed_r2": reference.score,
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
    import tempfile
    from time import perf_counter

    import torch

    from src.artifacts import save_predictor, write_json
    from src.ensemble_run import config_fingerprint
    from src.monitoring import training_events
    from src.training.patrick import train_model
    from src.training.validation import PreparedValidation

    claim(run_id, f"seed_{seed}")
    root, launch, config = load_launch(run_id)
    if seed not in config.ensemble.seeds:
        raise ValueError("unconfigured seed")
    rehearsal = json.loads((root / "rehearsal.json").read_text())
    if rehearsal["status"] != "passed" or rehearsal["code_sha256"] != launch["code_sha256"]:
        raise ValueError("GPU rehearsal required")
    directory = root / "seeds" / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tracker = tracker_for(run_id, f"seed-{seed:02d}", {**launch["config"], "seed": seed})
    tracker.define_metric("train/*", step_metric="train/batch")
    for key in ("mean_optimization_loss", "mean_unbalanced_loss", "responder_6_r2"):
        tracker.define_metric(f"train/epoch_{key}", step_metric="train/epoch")
    tracker.define_metric("val/*", step_metric="val/epoch")
    start = perf_counter()
    logged = int(tracker.summary.get("last_checkpoint_step", -1))
    try:
        prepared = prepared_for(root, launch, config)
        checkpoint = directory / "training.pt"

        def progress(record):
            nonlocal logged
            if not record["checkpoint_written"]:
                return
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for event in training_events(
                saved, days_per_epoch=len(prepared.dates), epochs=config.training.epochs
            ):
                if event["step"] > logged:
                    tracker.log(event["metrics"], step=event["step"])
                    logged = event["step"]
            tracker.summary["last_checkpoint_step"] = logged
            write_json(directory / "history.json", saved["history"])
            state = {
                **record,
                "seed": seed,
                "wall_seconds": perf_counter() - start,
                "wandb_url": tracker.url,
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
            progress=progress,
            validation=PreparedValidation(root / "validation_cache"),
        )
        if not (directory / "checkpoint").exists():
            with tempfile.TemporaryDirectory(dir=directory, prefix=".final-") as temporary:
                staged = Path(temporary) / "checkpoint"
                save_predictor(
                    staged, [model], prepared.features, prepared.scaler, config.online, [seed]
                )
                staged.rename(directory / "checkpoint")
        result = {
            "status": "complete",
            "seed": seed,
            "epochs": config.training.epochs,
            "config_sha256": config_fingerprint(config),
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
        output_volume.commit()
        tracking_volume.commit()


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
def replay_trial(run_id, name):
    import torch

    from src.cv import TemporalFold
    from src.online_sweep import ReplayDayCache, Trial, run_trial

    claim(run_id, f"trial_{name}")
    root, launch, config = load_launch(run_id)
    trial = next(Trial(**t) for t in launch["trials"] if t["name"] == name)
    fold = TemporalFold(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in launch["splits"].items()}
    )
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    def progress(record):
        if "completed_date" in record or record.get("complete"):
            output_volume.commit()
            print(json.dumps({"trial": name, **record}), flush=True)

    try:
        return run_trial(
            ReplayDayCache(root / "replay_days"),
            root / "initial_checkpoint",
            root / "trials" / name,
            trial,
            fold,
            device=config.training.device,
            provenance={
                "run_id": run_id,
                "code_sha256": launch["code_sha256"],
                "dataset_sha256": launch["dataset_sha256"],
            },
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
    scaledown_window=2,
    retries=1,
    volumes=volumes,
    secrets=[secret],
    include_source=False,
)
def coordinate(run_id, digest, code_sha, commit):
    import time

    import wandb

    from src.artifacts import write_json
    from src.config import load_config
    from src.cv import configured_folds
    from src.ensemble_run import assemble_ensemble, canonical, config_fingerprint
    from src.monitoring import evaluation_events
    from src.online_sweep import summarize_trials, trial_grid
    from src.safety import source_fingerprint

    claim(run_id, "controller")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid dataset fingerprint")
    if source_fingerprint() != code_sha:
        raise ValueError("local/deployed source differs")
    config = load_config(CONFIG)
    (fold,) = configured_folds(range(1380), config.cv)
    root = Path("/sweep/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    launch = canonical(
        {
            "run_id": run_id,
            "dataset_sha256": digest,
            "code_sha256": code_sha,
            "source_commit": commit,
            "config": config.to_dict(),
            "config_sha256": config_fingerprint(config),
            "splits": asdict(fold),
            "trials": [asdict(t) for t in trial_grid()],
            "max_training_gpus": 3,
            "max_replay_gpus": 4,
        }
    )
    if (root / "launch.json").exists() and json.loads((root / "launch.json").read_text()) != launch:
        raise ValueError("launch identity mismatch")
    write_json(root / "launch.json", launch)
    output_volume.commit()
    tracker = tracker_for(run_id, "overview", launch)
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

    def state(phase, **values):
        tracker.summary["phase"] = phase
        write_json(root / "status.json", {"phase": phase, "wandb_url": tracker.url, **values})
        output_volume.commit()

    def wait(call_id):
        while True:
            try:
                return modal.FunctionCall.from_id(call_id).get(timeout=0)
            except TimeoutError:
                time.sleep(30)

    try:
        state("preparing")
        wait(submit("prepare", prepare))
        state("gpu_rehearsal")
        wait(submit("rehearse", rehearse))
        for seed in config.ensemble.seeds:
            submit(f"seed_{seed}", train_seed, seed)
        while True:
            output_volume.reload()
            complete = [
                s
                for s in config.ensemble.seeds
                if (root / "seeds" / str(s) / "result.json").exists()
            ]
            state("training", completed_seeds=complete)
            tracker.log({"training/completed_seeds": len(complete)})
            if len(complete) == len(config.ensemble.seeds):
                break
            for s in config.ensemble.seeds:
                if s not in complete:
                    try:
                        modal.FunctionCall.from_id(calls[f"seed_{s}"]).get(timeout=0)
                    except TimeoutError:
                        pass
            time.sleep(30)
        assemble_ensemble(
            [root / "seeds" / str(s) for s in config.ensemble.seeds],
            config,
            root / "initial_checkpoint",
        )
        output_volume.commit()
        trials = trial_grid()
        for trial in trials:
            submit(f"trial_{trial.name}", replay_trial, trial.name)
            tracker.define_metric(f"{trial.name}/*", step_metric=f"{trial.name}/date_id")
        seen = {t.name: int(tracker.summary.get(f"{t.name}/days_completed", 0)) for t in trials}

        def observe():
            for t in trials:
                for event in evaluation_events(
                    root / "trials" / t.name,
                    t.mode,
                    replay_dates=fold.replay_dates,
                    total_batches=0,
                ):
                    m = event["metrics"]
                    if m[f"{t.mode}/days_completed"] > seen[t.name]:
                        tracker.log(
                            {
                                f"{t.name}/date_id": m["eval/date_id"],
                                **{
                                    f"{t.name}/{k.split('/', 1)[1]}": v
                                    for k, v in m.items()
                                    if k.startswith(t.mode + "/")
                                },
                            }
                        )
                        seen[t.name] = m[f"{t.mode}/days_completed"]

        while True:
            output_volume.reload()
            observe()
            finished = True
            for t in trials:
                try:
                    modal.FunctionCall.from_id(calls[f"trial_{t.name}"]).get(timeout=0)
                except TimeoutError:
                    finished = False
            state("replay", days_completed=seen)
            if finished:
                break
            time.sleep(30)
        output_volume.reload()
        observe()
        result = summarize_trials(root / "trials", trials, fold)
        tracker.log(
            {
                "comparison/rolling_r2": wandb.Image(str(root / "trials/rolling_comparison.png")),
                "comparison/results": wandb.Table(
                    columns=["trial", "r2", "delta_r2"],
                    data=[[r["name"], r["r2"], r["delta_r2"]] for r in result["trials"]],
                ),
            }
        )
        for r in result["trials"]:
            tracker.summary[f"{r['name']}/final_r2"] = r["r2"]
        state("complete")
        tracker.summary["status"] = "complete"
        tracker.finish()
        return {"status": "complete", "wandb_url": tracker.url}
    except Exception as error:
        write_json(
            root / "controller_failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        state("failed")
        tracker.finish(exit_code=1)
        raise
    finally:
        output_volume.commit()
        tracking_volume.commit()
