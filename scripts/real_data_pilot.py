"""Bounded real-data pilot: one fit, matched replays, and future-label perturbation.

Prepare locally: python -m scripts.real_data_pilot prepare --output artifacts/pilot-input.parquet
Run: python -m scripts.real_data_pilot run --data FILE --output NEW_DIR --device cpu
"""

import argparse
import gc
import importlib.metadata
import platform
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch

from src.artifacts import load_predictor, save_predictor, sha256_file, write_json
from src.config import load_config
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.data.schema import FEATURES, KEYS, RESPONDERS, validate_test
from src.experiment import PredictionWriter
from src.safety import safety_gate
from src.training.dispatch import fit, prepare

TRAIN = tuple(range(700, 704))
WARMUP = (704, 705)
SCORED = (706, 707, 708)
DATES = TRAIN + WARMUP + SCORED


def validate_panel(frame):
    if tuple(sorted(frame["date_id"].unique().to_list())) != DATES:
        raise ValueError("pilot input must contain exactly dates 700–708; holdout forbidden")
    return FrameSource(frame)


def prepare_slice(source, output):
    gate = safety_gate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = Path(source)
    paths = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
    selected = []
    for path in paths:
        scan = pl.scan_parquet(path)
        low, high = (
            scan.select(pl.col("date_id").min(), pl.col("date_id").max().alias("max"))
            .collect()
            .row(0)
        )
        if low <= DATES[-1] and high >= DATES[0]:
            selected.append(path)
    if not selected:
        raise ValueError("no partitions overlap the pilot interval")
    frame = pl.scan_parquet(selected).filter(pl.col("date_id").is_in(DATES)).collect().sort(KEYS)
    validate_panel(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    write_json(
        output.with_suffix(".json"),
        {
            "dates": DATES,
            "rows": frame.height,
            "sha256": sha256_file(output),
            "source_partitions": [str(p.resolve()) for p in selected],
            "safety_gate": gate,
            "note": "Only pilot rows exported; dates 1499–1698 excluded.",
        },
    )
    print(f"Prepared {frame.height:,} rows at {output}", flush=True)


def weights_digest(model):
    import hashlib

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def replay(frame, checkpoint, directory, device, online):
    directory.mkdir()
    source = validate_panel(frame)
    predictor = load_predictor(checkpoint, device)
    predictor.config = replace(predictor.config, enabled=online)
    initial_digest = weights_digest(predictor.models[0])
    mean, scale = predictor.scaler.mean.copy(), predictor.scaler.scale.copy()
    writer = PredictionWriter(directory / "predictions")
    releases, timings, daily = [], [], []
    last_date = None

    def audited_predict(test, lags):
        nonlocal last_date
        validate_test(test)
        date, time = int(test["date_id"][0]), int(test["time_id"][0])
        if date != last_date:
            print(f"  replay online={online}: date {date}", flush=True)
            last_date = date
        before = len(predictor.update_log)
        if time == 0:
            assert lags is not None
            # Independent evaluator-side check; only the supplied API lag frame is
            # passed to the actual predictor. No truth lookup reaches the model.
            previous = frame.filter(pl.col("date_id") == date - 1)
            expected = (
                previous.select(KEYS + RESPONDERS)
                .with_columns(pl.lit(date).cast(previous["date_id"].dtype).alias("date_id"))
                .rename({r: r + "_lag_1" for r in RESPONDERS})
            )
            assert lags.sort(KEYS).equals(expected.sort(KEYS))
            releases.append(
                {"date": date, "time": time, "source_date": date - 1, "rows": lags.height}
            )
        else:
            assert lags is None
        if device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        result = predictor.predict(test, lags)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(perf_counter() - start)
        if len(predictor.update_log) != before:
            update = predictor.update_log[-1]
            assert time == 0 and update["source_date"] == date - 1
            assert update["rows"] == releases[-1]["rows"]
        return result

    def sink(batch):
        writer(batch)
        if not daily or daily[-1]["date"] != int(batch["date_id"][0]):
            daily.append({"date": int(batch["date_id"][0]), "rows": 0, "calls": 0})
        daily[-1]["rows"] += batch.height
        daily[-1]["calls"] += 1

    start = perf_counter()
    result = APISimulator(source, WARMUP + SCORED, scored_dates=SCORED).run(
        audited_predict, collect_predictions=False, prediction_sink=sink
    )
    writer.flush()
    seconds = perf_counter() - start
    assert np.array_equal(mean, predictor.scaler.mean)
    assert np.array_equal(scale, predictor.scaler.scale)
    expected_updates = list(range(WARMUP[0] + 1, SCORED[-1] + 1)) if online else []
    assert [u["released_at"][0] for u in predictor.update_log] == expected_updates
    assert result.metric.rows == frame.filter(pl.col("date_id").is_in(SCORED)).height
    final_digest = weights_digest(predictor.models[0])
    assert (initial_digest != final_digest) == online
    summary = {
        "online": online,
        "seconds": seconds,
        "calls": result.calls,
        "score": result.metric.score,
        "sse": result.metric.sse,
        "denominator": result.metric.denominator,
        "scored_rows": result.metric.rows,
        "call_seconds_p50_p95_max": np.quantile(timings, [0.5, 0.95, 1.0]).tolist(),
        "initial_weights_sha256": initial_digest,
        "final_weights_sha256": final_digest,
        "updates": predictor.update_log,
        "lag_releases": releases,
        "daily": daily,
        "scaler_frozen": True,
    }
    write_json(directory / "result.json", summary)
    predictor = None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_pilot(data, output, device="cuda", method="grigoreva"):
    gate = safety_gate()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = pl.read_parquet(data).sort(KEYS)
    source = validate_panel(frame)
    if method not in {"grigoreva", "patrick"}:
        raise ValueError("unknown pilot method")
    filename = "patrick.yaml" if method == "patrick" else "real_data_pilot.yaml"
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    config = replace(
        config,
        training=replace(config.training, device=device, epochs=1),
        cv=replace(config.cv, min_date=700, gap_days=2, validation_days=3, n_splits=1),
    )
    write_json(output / "config.json", config.to_dict())
    write_json(output / "safety_gate.json", gate)
    write_json(output / "splits.json", {"train": TRAIN, "warmup": WARMUP, "scored": SCORED})
    write_json(
        output / "data_manifest.json",
        {
            "sha256": sha256_file(data),
            "dates": DATES,
            "rows": frame.height,
            "daily": frame.group_by("date_id")
            .agg(
                pl.len().alias("rows"),
                pl.col("time_id").n_unique().alias("timestamps"),
                pl.col("symbol_id").unique().sort().alias("symbols"),
            )
            .sort("date_id")
            .to_dicts(),
            "feature_null_counts": frame.select(FEATURES).null_count().to_dicts()[0],
        },
    )
    started = perf_counter()
    print("Preparing training dates 700–703", flush=True)
    prepared = prepare(config, source, TRAIN, output / "training_cache")
    preparation_seconds = perf_counter() - started
    print(f"Fitting {method}: published dimensions, seed 0, one epoch", flush=True)
    start = perf_counter()
    model, history = fit(config, prepared, config.model, 0)
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = perf_counter() - start
    save_predictor(
        output / "checkpoint", [model], prepared.features, prepared.scaler, config.online, [0]
    )
    write_json(output / "training_history.json", history)
    del model, prepared
    gc.collect()
    off = replay(frame, output / "checkpoint", output / "offline", device, False)
    on = replay(frame, output / "checkpoint", output / "online", device, True)
    # Final replay day's labels cannot be consumed: they would arrive on day 709.
    poisoned = frame.with_columns(
        [
            pl.when(pl.col("date_id") == SCORED[-1])
            .then(pl.col(r) * -17 + 123)
            .otherwise(pl.col(r))
            .alias(r)
            for r in RESPONDERS
        ]
    )
    poison = replay(poisoned, output / "checkpoint", output / "future_labels_changed", device, True)
    for date in WARMUP + SCORED:
        normal = pl.read_parquet(output / "online/predictions" / f"date_{date}.parquet")
        changed = pl.read_parquet(
            output / "future_labels_changed/predictions" / f"date_{date}.parquet"
        )
        assert normal.equals(changed), f"future labels changed predictions on {date}"
    assert on["final_weights_sha256"] == poison["final_weights_sha256"]
    first = f"date_{WARMUP[0]}.parquet"
    assert pl.read_parquet(output / "offline/predictions" / first).equals(
        pl.read_parquet(output / "online/predictions" / first)
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "status": "passed",
        "method": method,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "polars", "numpy")},
        "preparation_seconds": preparation_seconds,
        "training_seconds": training_seconds,
        "total_seconds": perf_counter() - started,
        "process_peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "offline": off,
        "online": on,
        "checks": {
            "same_initial_checkpoint": off["initial_weights_sha256"]
            == on["initial_weights_sha256"],
            "future_responder_perturbation_predictions_identical": True,
            "future_responder_perturbation_final_weights_identical": True,
            "first_warmup_day_predictions_identical": True,
        },
        "score_note": "Execution pilot only; four training days and one epoch are not a performance comparison.",
    }
    write_json(output / "result.json", summary)
    print(f"Pilot passed in {summary['total_seconds']:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--data", default="data/competition/train.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--method", choices=["grigoreva", "patrick"], default="grigoreva")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_slice(args.data, args.output)
    else:
        run_pilot(args.data, args.output, args.device, args.method)
