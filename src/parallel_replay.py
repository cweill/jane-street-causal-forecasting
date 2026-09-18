"""Independent, resumable replay modes using one immutable initial checkpoint.

Each worker owns only its mode directory. The existing sequential reproduction is
retained as the reference implementation for equivalence tests and benchmarks.
"""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import polars as pl

from src.artifacts import load_predictor, sha256_file, write_json
from src.data.api_simulator import APISimulator
from src.data.loader import CachedDaySource, RestrictedDateSource
from src.plotting import plot_online_comparison, score_day
from src.reproduction import weights_fingerprint
from src.training.checkpoints import load_replay_checkpoint, save_replay_checkpoint


def import_frozen_prefix(source, destination, checkpoint, dates, scored, signature, initial_hash):
    """Copy an immutable committed prefix; publish its completion marker last.

    Only the latest imported date needs full model state. Earlier markers retain
    their source location so their predictions/metrics can be audited there.
    """
    source = Path(source)
    record_path = destination / "prefix_import.json"
    if record_path.exists():
        return json.loads(record_path.read_text())["days"]
    original = source.parent
    splits = json.loads((original / "splits.json").read_text())
    if (
        tuple(splits["warmup_dates"] + splits["validation_dates"]) != dates
        or tuple(splits["validation_dates"]) != scored
    ):
        raise ValueError("imported frozen split identity mismatch")
    for name in ("weights.pt", "metadata.json"):
        if sha256_file(original / "initial_checkpoint" / name) != sha256_file(checkpoint / name):
            raise ValueError("imported initial checkpoint identity mismatch")
    old_signature = json.loads((original / "identity.json").read_text())["signature"]
    committed = []
    for date in dates:
        marker = source / "checkpoints" / f"date_{date}" / "replay.json"
        if not marker.exists():
            break
        state = json.loads(marker.read_text())
        if state["signature"] != old_signature or state["completed_date"] != date:
            raise ValueError("imported replay identity/date mismatch")
        committed.append((date, state))
    if committed:
        latest = source / "checkpoints" / f"date_{committed[-1][0]}"
        verified, _ = load_replay_checkpoint(latest, signature=old_signature, device="cpu")
        if verified.config.enabled or weights_fingerprint(verified.models[0]) != initial_hash:
            raise ValueError("import source is not the matching frozen model")
        del verified
        for date, state in committed:
            day = destination / f"date_{date}"
            day.mkdir(exist_ok=True)
            record = json.loads((source / f"date_{date}" / "result.json").read_text())
            if record["date_id"] != date:
                raise ValueError("imported daily result date mismatch")
            write_json(day / "result.json", record)
            shutil.copyfile(source / f"date_{date}.parquet", destination / f"date_{date}.parquet")
            target = destination / "checkpoints" / f"date_{date}"
            target.mkdir(parents=True, exist_ok=True)
            if date == committed[-1][0]:
                shutil.copytree(latest, target, dirs_exist_ok=True)
            write_json(
                target / "replay.json",
                {
                    **state,
                    "signature": signature,
                    "imported_from": str(source),
                    "original_signature": old_signature,
                },
            )
    write_json(
        record_path,
        {
            "source": str(source),
            "days": len(committed),
            "last_date": committed[-1][0] if committed else None,
        },
    )
    return len(committed)


def run_replay_mode(
    source,
    directory,
    *,
    checkpoint,
    mode,
    replay_dates,
    scored_dates,
    device,
    provenance,
    fast=True,
    progress=None,
    source_prefix=None,
):
    if mode not in ("offline", "online"):
        raise ValueError("invalid replay mode")
    if source_prefix is not None and mode != "offline":
        raise ValueError("only frozen replay supports prefix import")
    dates, scored = tuple(replay_dates), tuple(scored_dates)
    if (
        not dates
        or dates != tuple(range(dates[0], dates[-1] + 1))
        or not scored
        or not set(scored).issubset(dates)
    ):
        raise ValueError("consecutive replay dates and nonempty scored subset required")
    checkpoint, directory = Path(checkpoint), Path(directory)
    identity = {
        "mode": mode,
        "replay_dates": dates,
        "scored_dates": scored,
        "fast": fast,
        "checkpoint_weights_sha256": sha256_file(checkpoint / "weights.pt"),
        "checkpoint_metadata_sha256": sha256_file(checkpoint / "metadata.json"),
        "provenance": provenance,
        "source_prefix": str(source_prefix) if source_prefix is not None else None,
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("replay identity mismatch")
    else:
        if any(directory.iterdir()):
            raise ValueError("nonempty replay directory without identity")
        write_json(manifest, {**identity, "signature": signature})

    def report(**record):
        write_json(directory / "status.json", {"phase": mode, **record})
        if progress:
            progress({"phase": mode, **record})

    available = set(source.dates())
    source = RestrictedDateSource(source, [d for d in (dates[0] - 1, *dates) if d in available])
    if fast:
        source = CachedDaySource(source)
    initial = load_predictor(checkpoint, device)
    initial_hash = weights_fingerprint(initial.models[0])
    reused_days = 0
    if source_prefix is not None:
        reused_days = import_frozen_prefix(
            source_prefix, directory, checkpoint, dates, scored, signature, initial_hash
        )
        report(reused_days=reused_days)
    snapshots = directory / "checkpoints"
    records, offset, last = [], 0, None
    for date in dates:
        marker = snapshots / f"date_{date}" / "replay.json"
        if not marker.exists():
            break
        state = json.loads(marker.read_text())
        if state["signature"] != signature or state["completed_date"] != date:
            raise ValueError("replay checkpoint identity/date mismatch")
        records.append(json.loads((directory / f"date_{date}" / "result.json").read_text()))
        last = date
    if last is None:
        predictor = initial
        predictor.config = replace(predictor.config, enabled=mode == "online")
    else:
        del initial
        predictor, state = load_replay_checkpoint(
            snapshots / f"date_{last}", signature=signature, device=device
        )
        offset = state["row_offset"]
    if predictor.config.enabled != (mode == "online"):
        raise ValueError("checkpoint replay mode mismatch")
    predictor.fast_inference = fast
    for date in dates[len(records) :]:
        report(starting_date=date, replay_days_completed=len(records))
        started = perf_counter()
        result = APISimulator(
            source, [date], scored_dates=scored, row_offset=offset, prepartition_truth=fast
        ).run(predictor.predict)
        offset += result.predictions.height
        day_directory = directory / f"date_{date}"
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
        report(completed_date=date, replay_days_completed=len(records))
    sse = sum(r["primary"]["sse"] for r in records)
    energy = sum(r["primary"]["denominator"] for r in records)
    if energy <= 0:
        raise ValueError("primary score has zero denominator")
    final_hash = weights_fingerprint(predictor.models[0])
    if mode == "offline" and final_hash != initial_hash:
        raise ValueError("frozen replay mutated weights")
    for update in predictor.update_log:
        if update["source_date"] != update["released_at"][0] - 1:
            raise ValueError("online update used unavailable labels")
    summary = {
        "score": 1 - sse / energy,
        "sse": sse,
        "denominator": energy,
        "scored_rows": sum(r["primary"]["rows"] for r in records),
        "initial_weights_sha256": initial_hash,
        "final_weights_sha256": final_hash,
        "updates": predictor.update_log,
        "seconds": sum(r["seconds"] for r in records),
        "reused_days": reused_days,
    }
    pl.DataFrame([r["diagnostic"] for r in records]).write_parquet(directory / "daily.parquet")
    write_json(directory / "result.json", summary)
    report(complete=True, replay_days_completed=len(records))
    return summary


def finish_comparison(directory, *, window=20, scored_start=None):
    directory = Path(directory)
    results = {
        mode: json.loads((directory / mode / "result.json").read_text())
        for mode in ("offline", "online")
    }
    if results["offline"]["initial_weights_sha256"] != results["online"]["initial_weights_sha256"]:
        raise ValueError("initial model mismatch between replays")
    identities = {
        mode: json.loads((directory / mode / "identity.json").read_text()) for mode in results
    }
    for key in (
        "checkpoint_weights_sha256",
        "checkpoint_metadata_sha256",
        "replay_dates",
        "scored_dates",
        "provenance",
    ):
        if identities["offline"][key] != identities["online"][key]:
            raise ValueError(f"paired replay identity mismatch: {key}")
    first = identities["offline"]["replay_dates"][0]
    if not pl.read_parquet(directory / "offline" / f"date_{first}.parquet").equals(
        pl.read_parquet(directory / "online" / f"date_{first}.parquet")
    ):
        raise ValueError("online replay changed first-day predictions before released labels")
    for suffix in ("png", "svg", "csv"):
        (directory / f"online_learning.{suffix}").unlink(missing_ok=True)
    paths = plot_online_comparison(
        pl.read_parquet(directory / "offline/daily.parquet"),
        pl.read_parquet(directory / "online/daily.parquet"),
        directory / "online_learning",
        title="Patrick reconstruction · matched parallel replays",
        window=window,
        scored_start=scored_start,
    )
    summary = {"status": "complete", **results, "plot": paths}
    write_json(directory / "result.json", summary)
    return summary
