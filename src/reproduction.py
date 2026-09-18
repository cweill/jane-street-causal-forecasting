"""One Patrick fit, matched API replays, restartable daily artifacts, and the OL plot."""

import hashlib
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import polars as pl

from src.artifacts import load_predictor, save_predictor, write_json
from src.cv import configured_folds
from src.data.api_simulator import APISimulator
from src.data.loader import RestrictedDateSource
from src.plotting import plot_online_comparison, score_day
from src.safety import safety_gate
from src.training.cache import prepare_cached
from src.training.checkpoints import load_replay_checkpoint, save_replay_checkpoint
from src.training.patrick import train_model


def weights_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_online_comparison(
    source, config, output, *, cache_root, dataset_sha256, resume=False, progress=None, window=20
):
    gate = safety_gate()
    if getattr(config, "method", None) != "patrick" or len(config.members) != 1:
        raise ValueError("the bounded plot run requires one Patrick model/seed")
    (fold,) = configured_folds(source.dates(), config.cv)
    identity = {
        "config": config.to_dict(),
        "dataset_sha256": dataset_sha256,
        "code_sha256": gate["code_and_tests_sha256"],
        "window": window,
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if json.loads((output / "identity.json").read_text())["signature"] != signature:
            raise ValueError("run identity mismatch; fork a new run instead of resuming")
    else:
        output.mkdir(parents=True)
        write_json(output / "identity.json", {"signature": signature, **identity})
        write_json(output / "splits.json", asdict(fold))
        write_json(output / "safety_gate.json", gate)
    source = RestrictedDateSource(source, fold.train_dates + fold.replay_dates)

    def report(phase, **values):
        record = {"phase": phase, **values}
        # Status is informational; checkpoints and day directories determine resume state.
        write_json(output / "status.json", record)
        if progress is not None:
            progress(record)

    checkpoint = output / "initial_checkpoint"
    if not checkpoint.exists():
        report("preparing")
        started = perf_counter()
        prepared, cache = prepare_cached(
            RestrictedDateSource(source, fold.train_dates),
            fold.train_dates,
            config.features,
            cache_root,
            dataset_sha256,
        )
        write_json(
            output / "preparation_cache.json", {**cache, "seconds": perf_counter() - started}
        )
        model, history = train_model(
            prepared,
            config.model,
            training=config.training,
            seed=config.members[0][1],
            checkpoint_path=output / "training.pt",
            progress=lambda record: report("training", **record),
        )
        with tempfile.TemporaryDirectory(prefix=".initial-", dir=output) as temp:
            staged = Path(temp) / "checkpoint"
            save_predictor(
                staged,
                [model],
                prepared.features,
                prepared.scaler,
                config.online,
                [config.members[0][1]],
            )
            staged.rename(checkpoint)
        write_json(output / "training_history.json", history)
        del model, prepared
    initial = load_predictor(checkpoint, config.training.device)
    initial_hash = weights_fingerprint(initial.models[0])
    del initial
    results = {}
    for mode, online in (("offline", False), ("online", True)):
        directory = output / mode
        directory.mkdir(exist_ok=True)
        snapshots = directory / "checkpoints"
        snapshots.mkdir(exist_ok=True)
        completed = sorted(
            int(p.name.removeprefix("date_")) for p in snapshots.glob("date_*") if p.is_dir()
        )
        if completed:
            predictor, state = load_replay_checkpoint(
                snapshots / f"date_{completed[-1]}",
                signature=signature,
                device=config.training.device,
            )
            offset = state["row_offset"]
            last = state["completed_date"]
        else:
            predictor = load_predictor(checkpoint, config.training.device)
            predictor.config = replace(predictor.config, enabled=online)
            offset, last = 0, None
        if predictor.config.enabled != online:
            raise ValueError("checkpoint belongs to the wrong replay mode")
        records = []
        for date in fold.replay_dates:
            day_directory = directory / f"date_{date}"
            if last is not None and date <= last:
                records.append(json.loads((day_directory / "result.json").read_text()))
                continue
            report(mode, starting_date=date)
            started = perf_counter()
            result = APISimulator(
                source, [date], scored_dates=fold.validation_dates, row_offset=offset
            ).run(predictor.predict)
            offset += result.predictions.height
            day_directory.mkdir(exist_ok=True)
            result.predictions.write_parquet(directory / f"date_{date}.parquet")
            record = {
                "date_id": date,
                "seconds": perf_counter() - started,
                "diagnostic": score_day(result.predictions, source.day(date)),
                "primary": {
                    "sse": result.metric.sse,
                    "denominator": result.metric.denominator,
                    "rows": result.metric.rows,
                },
                "calls": result.calls,
            }
            write_json(day_directory / "result.json", record)
            save_replay_checkpoint(
                predictor, snapshots / f"date_{date}", signature=signature, row_offset=offset
            )
            records.append(record)
            report(mode, completed_date=date, replay_days_completed=len(records))
        sse = sum(r["primary"]["sse"] for r in records)
        denominator = sum(r["primary"]["denominator"] for r in records)
        if denominator <= 0:
            raise ValueError("primary score has zero denominator")
        final_hash = weights_fingerprint(predictor.models[0])
        if not online and final_hash != initial_hash:
            raise ValueError("frozen replay changed weights")
        results[mode] = {
            "score": 1 - sse / denominator,
            "sse": sse,
            "denominator": denominator,
            "scored_rows": sum(r["primary"]["rows"] for r in records),
            "initial_weights_sha256": initial_hash,
            "final_weights_sha256": final_hash,
            "updates": predictor.update_log,
            "seconds": sum(r["seconds"] for r in records),
        }
        for update in predictor.update_log:
            if update["source_date"] != update["released_at"][0] - 1:
                raise ValueError("online update used labels before their release")
        pl.DataFrame([r["diagnostic"] for r in records]).write_parquet(directory / "daily.parquet")
        write_json(directory / "result.json", results[mode])
        del predictor
    first = fold.replay_dates[0]
    if not pl.read_parquet(output / "offline" / f"date_{first}.parquet").equals(
        pl.read_parquet(output / "online" / f"date_{first}.parquet")
    ):
        raise ValueError("online learning changed the first day before eligible labels")
    # If plotting was interrupted, only regenerate these derived files; model states stay fixed.
    for suffix in ("png", "svg", "csv"):
        path = output / f"online_learning.{suffix}"
        if path.exists():
            path.unlink()
    paths = plot_online_comparison(
        pl.read_parquet(output / "offline/daily.parquet"),
        pl.read_parquet(output / "online/daily.parquet"),
        output / "online_learning",
        title=f"Patrick reconstruction · 1 seed · {config.training.epochs} fixed epochs",
        window=window,
        scored_start=fold.validation_dates[0],
    )
    summary = {
        "status": "complete",
        **results,
        "plot": paths,
        "note": "Reconstruction assumptions are fixed; no claim of matching author scores.",
    }
    write_json(output / "result.json", summary)
    report("complete")
    return summary
