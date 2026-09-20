"""Four additional development replays; reuse original models, caches and baselines."""

import json
import re
from dataclasses import asdict
from pathlib import Path

import modal

from scripts.modal_ensemble_run import image
from scripts.modal_online_sweep import tracker_for

app = modal.App("patrick-online-refinement")
output = modal.Volume.from_name("janestreet-patrick-online-refinement", create_if_missing=True)
original_volume = modal.Volume.from_name("janestreet-patrick-ol-sweep")
tracking = modal.Volume.from_name("janestreet-wandb-monitor")
owners = modal.Dict.from_name("janestreet-refinement-calls", create_if_missing=True)
volumes = {
    "/refinement": output,
    "/original": original_volume.with_mount_options(read_only=True),
    "/tracking": tracking,
}
ORIGINAL = "ol-sweep-20260919T194601Z"
ORIGIN = Path("/original/runs") / ORIGINAL
secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


def claim(run_id, role):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    key, call = f"{run_id}/owner/{role}", modal.current_function_call_id()
    if not owners.put(key, call, skip_if_exists=True) and owners[key] != call:
        raise ValueError("another call owns role")


def read(path):
    return json.loads(Path(path).read_text())


def load(run_id):
    from src.cv import TemporalFold
    from src.safety import source_fingerprint

    output.reload()
    root = Path("/refinement/runs") / run_id
    launch = read(root / "launch.json")
    if source_fingerprint() != launch["code_sha256"]:
        raise ValueError("deployment differs from launch")
    fold = TemporalFold(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in launch["splits"].items()}
    )
    return root, launch, fold


def verify_runtime_archive():
    import ast
    import tarfile

    root = Path(__file__).resolve().parents[1]
    files = [
        root / "uv.lock",
        *[
            p
            for folder in ("data", "models", "training")
            for p in (root / "src" / folder).rglob("*.py")
        ],
        *[
            root / "src" / f"{name}.py"
            for name in (
                "metric",
                "parallel_replay",
                "artifacts",
                "config",
                "plotting",
                "reproduction",
            )
        ],
    ]
    with tarfile.open(ORIGIN / "source.tar.gz", "r:gz") as archive:
        for p in files:
            name = str(p.relative_to(root))
            if archive.extractfile(name).read() != p.read_bytes():
                raise ValueError(f"reused replay runtime differs: {name}")
        before = ast.parse(archive.extractfile("src/online_sweep.py").read())
        after = ast.parse((root / "src/online_sweep.py").read_text())
        for name in ("Trial", "ReplayDayCache", "validate_fold", "trial_checkpoint", "run_trial"):
            get = lambda tree, n=name: ast.dump(
                next(node for node in tree.body if getattr(node, "name", None) == n)
            )
            if get(before) != get(after):
                raise ValueError(f"reused replay runtime differs: {name}")
    return {"passed": True, "files": len(files), "replay_definitions": 5}


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
def replay(run_id, name):
    import polars as pl
    import torch

    from src.artifacts import write_json
    from src.online_refinement import new_trials
    from src.online_sweep import ReplayDayCache, run_trial

    claim(run_id, name)
    root, launch, fold = load(run_id)
    if not read(root / "ready.json")["passed"]:
        raise ValueError("preflight required")
    trial = next(t for t in new_trials() if t.name == name)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    directory = root / "trials" / name
    first = directory / "online" / f"date_{fold.replay_dates[0]}.parquet"

    def check():
        frozen = ORIGIN / "trials/frozen/offline" / first.name
        if not pl.read_parquet(first).equals(pl.read_parquet(frozen)):
            raise ValueError("first-day predictions differ from reused frozen baseline")
        write_json(
            directory / "first_day_parity.json", {"passed": True, "date_id": fold.replay_dates[0]}
        )

    if first.exists():
        check()

    def progress(record):
        if record.get("completed_date") == fold.replay_dates[0]:
            check()
        if "completed_date" in record or record.get("complete"):
            output.commit()
            print(json.dumps({"trial": name, **record}), flush=True)

    try:
        return run_trial(
            ReplayDayCache(ORIGIN / "replay_days"),
            ORIGIN / "initial_checkpoint",
            directory,
            trial,
            fold,
            device="cuda",
            provenance=launch["new_provenance"],
            progress=progress,
        )
    except Exception as error:
        write_json(
            directory / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        output.commit()


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=28800,
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
    from src.cv import TemporalFold
    from src.monitoring import evaluation_events
    from src.online_refinement import block_comparison, new_trials, reused_trials
    from src.online_sweep import ReplayDayCache, summarize_trials, trial_checkpoint
    from src.safety import safety_gate, source_fingerprint

    claim(run_id, "controller")
    if source_fingerprint() != code_sha:
        raise ValueError("local/deployed source mismatch")
    root = Path("/refinement/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    origin = read(ORIGIN / "launch.json")
    fold = TemporalFold(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in origin["splits"].items()}
    )
    old_prov = {
        "run_id": ORIGINAL,
        "code_sha256": origin["code_sha256"],
        "dataset_sha256": origin["dataset_sha256"],
    }
    new_prov = {**old_prov, "run_id": run_id, "code_sha256": code_sha}
    trials = [*reused_trials(), *new_trials()]
    launch = {
        "run_id": run_id,
        "code_sha256": code_sha,
        "source_commit": commit,
        "original_run": ORIGINAL,
        "original_code_sha256": origin["code_sha256"],
        "dataset_sha256": origin["dataset_sha256"],
        "splits": origin["splits"],
        "new_trials": [asdict(t) for t in new_trials()],
        "reused_trials": [asdict(t) for t in reused_trials()],
        "new_provenance": new_prov,
        "max_gpus": 4,
        "offline_training": False,
    }
    if (root / "launch.json").exists() and read(root / "launch.json") != launch:
        raise ValueError("launch identity mismatch")
    write_json(root / "launch.json", launch)
    output.commit()
    tracker = tracker_for(run_id, "overview", launch)

    def state(phase, **values):
        tracker.summary["phase"] = phase
        write_json(root / "status.json", {"phase": phase, "wandb_url": tracker.url, **values})
        output.commit()

    try:
        state("preflight")
        gate = safety_gate()
        if gate["code_and_tests_sha256"] != code_sha:
            raise ValueError("gate/source mismatch")
        write_json(root / "safety_gate.json", gate)
        output.commit()
        if (
            read(ORIGIN / "status.json")["phase"] != "complete"
            or not read(ORIGIN / "safety_gate.json")["passed"]
        ):
            raise ValueError("original verified study must be complete")
        runtime = verify_runtime_archive()
        write_json(root / "runtime_parity.json", runtime)
        if (
            fold.train_dates != tuple(range(1060))
            or fold.warmup_dates != tuple(range(1060, 1180))
            or fold.validation_dates != tuple(range(1180, 1380))
            or read(ORIGIN / "initial_checkpoint/metadata.json")["seeds"] != [0, 1, 2]
        ):
            raise ValueError("fixed three-seed development protocol required")
        state("verifying_cached_days")
        cache = ReplayDayCache(ORIGIN / "replay_days")
        if cache.dates() != (1059, *fold.replay_dates):
            raise ValueError("cached source coverage mismatch")
        for date in cache.dates():
            cache.day(date)
        for t in new_trials():
            trial_checkpoint(
                ORIGIN / "initial_checkpoint",
                root / "trials" / t.name / "initial_checkpoint",
                t,
                fold,
            )
        roots = {
            t.name: (ORIGIN if t in reused_trials() else root) / "trials" / t.name for t in trials
        }
        provenances = {t.name: old_prov if t in reused_trials() else new_prov for t in trials}
        # Validate both completed references before incurring GPU cost.
        summarize_trials(
            root / "reference_check",
            reused_trials(),
            fold,
            trial_directories={t.name: roots[t.name] for t in reused_trials()},
            expected_provenance={t.name: old_prov for t in reused_trials()},
        )
        write_json(root / "ready.json", {"passed": True, "runtime_parity": runtime})
        output.commit()
        calls = {}
        for t in new_trials():
            key = f"{run_id}/call/{t.name}"
            call = owners.get(key) or owners.get(f"{run_id}/owner/{t.name}")
            if not call:
                call = replay.spawn(run_id, t.name).object_id
                owners[key] = call
            calls[t.name] = call
            write_json(root / "calls.json", calls)
            output.commit()
        for t in trials:
            tracker.define_metric(f"{t.name}/*", step_metric=f"{t.name}/date_id")
        seen = {t.name: int(tracker.summary.get(f"{t.name}/days_completed", 0)) for t in trials}
        while True:
            output.reload()
            for t in trials:
                for event in evaluation_events(
                    roots[t.name], t.mode, replay_dates=fold.replay_dates, total_batches=0
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
            state("replaying", days_completed=seen)
            done = True
            for call in calls.values():
                try:
                    modal.FunctionCall.from_id(call).get(timeout=0)
                except TimeoutError:
                    done = False
            if done and all(v == len(fold.replay_dates) for v in seen.values()):
                break
            time.sleep(30)
        output.reload()
        result = summarize_trials(
            root / "comparison",
            trials,
            fold,
            trial_directories=roots,
            expected_provenance=provenances,
        )
        blocks = block_comparison(roots, trials, fold)
        blocks.write_csv(root / "comparison/scored_blocks.csv")
        tracker.log(
            {
                "comparison/rolling_r2": wandb.Image(
                    str(root / "comparison/rolling_comparison.png")
                ),
                "comparison/results": wandb.Table(
                    columns=["trial", "r2", "delta_r2"],
                    data=[[r["name"], r["r2"], r["delta_r2"]] for r in result["trials"]],
                ),
                "comparison/scored_blocks": wandb.Table(columns=blocks.columns, data=blocks.rows()),
            }
        )
        for r in result["trials"]:
            tracker.summary[f"{r['name']}/final_r2"] = r["r2"]
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
