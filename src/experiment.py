"""End-to-end temporal CV: each fold fits alone and is evaluated through API replay."""

import importlib.metadata
import platform
from dataclasses import asdict
from pathlib import Path

import polars as pl

from src.artifacts import load_predictor, save_predictor, write_json
from src.cv import temporal_folds
from src.data.api_simulator import APISimulator
from src.safety import safety_gate
from src.training.dispatch import fit, prepare


class PredictionWriter:
    """Flush predictions daily to bound memory independently of validation length."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.date, self.parts = None, []

    def __call__(self, frame):
        date = int(frame["date_id"][0])
        if self.date is not None and date != self.date:
            self.flush()
        self.date = date
        self.parts.append(frame)

    def flush(self):
        if self.parts:
            pl.concat(self.parts).write_parquet(self.directory / f"date_{self.date}.parquet")
            self.parts = []


def source_manifest(source):
    if hasattr(source, "path"):
        path = source.path
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        return {
            "kind": "parquet",
            "files": [
                {
                    "path": str(p.resolve()),
                    "bytes": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
                for p in files
            ],
            "identity_note": "File metadata only; retain immutable input files for reproducibility.",
        }
    return {"kind": "in_memory", "dates": list(source.dates())}


def run_experiment(source, config, output):
    gate = safety_gate()
    return _run_verified(source, config, output, gate)


def _run_verified(source, config, output, gate):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "safety_gate.json", gate)
    write_json(output / "config.json", config.to_dict())
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ("numpy", "polars", "torch", "PyYAML")},
        },
    )
    write_json(output / "data_manifest.json", source_manifest(source))
    cv = asdict(config.cv)
    min_date = cv.pop("min_date")
    dates = [d for d in source.dates() if d >= min_date]
    folds = temporal_folds(dates, **cv)
    write_json(output / "splits.json", [asdict(f) for f in folds])
    results = []
    for fold in folds:
        directory = output / f"fold_{fold.index}"
        directory.mkdir()
        print(
            f"fold {fold.index}: train {fold.train_dates[0]}..{fold.train_dates[-1]}, "
            f"validate {fold.validation_dates[0]}..{fold.validation_dates[-1]}",
            flush=True,
        )
        prepared = prepare(config, source, fold.train_dates, directory / "training_cache")
        models, seeds, histories = [], [], []
        for model_config, seed in config.members:
            label = getattr(model_config, "architecture", "patrick")
            print(f"  fitting {label}, seed {seed}", flush=True)
            model, history = fit(config, prepared, model_config, seed)
            models.append(model)
            seeds.append(seed)
            histories.append({"model": asdict(model_config), "seed": seed, "history": history})
        write_json(directory / "training_history.json", histories)
        save_predictor(
            directory / "checkpoint",
            models,
            prepared.features,
            prepared.scaler,
            config.online,
            seeds,
        )
        # Load the saved checkpoint for replay; verifies artifact completeness and isolates state.
        predictor = load_predictor(directory / "checkpoint", config.training.device)
        writer = PredictionWriter(directory / "predictions")
        result = APISimulator(source, fold.replay_dates, scored_dates=fold.validation_dates).run(
            predictor.predict, collect_predictions=False, prediction_sink=writer
        )
        writer.flush()
        summary = {
            "fold": fold.index,
            "score": result.metric.score,
            "sse": result.metric.sse,
            "denominator": result.metric.denominator,
            "scored_rows": result.metric.rows,
            "calls": result.calls,
            "max_call_seconds": result.max_call_seconds,
            "online_updates": len(predictor.update_log),
        }
        write_json(directory / "updates.json", predictor.update_log)
        write_json(directory / "result.json", summary)
        results.append(summary)
        print(f"  weighted zero-mean R²: {result.metric.score:.6f}", flush=True)
    # Folds have disjoint validation intervals. Pool sufficient statistics rather than scores.
    total_sse = sum(r["sse"] for r in results)
    total_denominator = sum(r["denominator"] for r in results)
    summary = {
        "name": config.name,
        "pooled_score": 1 - total_sse / total_denominator,
        "folds": results,
        "score_note": "Research CV only; no claim of reproduced leaderboard performance.",
    }
    write_json(output / "result.json", summary)
    return summary
