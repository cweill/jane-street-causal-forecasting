"""Atomic training state and day-boundary replay snapshots for Patrick."""

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import polars as pl
import torch

from src.artifacts import load_predictor, save_predictor, sha256_file, write_json


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_signature(prepared, model_config, training, seed):
    from src.training.cache import preprocessing_fingerprint

    identity = {
        "model": asdict(model_config),
        "training": asdict(training),
        "seed": seed,
        "features": prepared.features.state_dict(),
        "dates": prepared.dates,
        "data": {str(d): sha256_file(prepared.directory / f"{d}.npz") for d in prepared.dates},
        "code": preprocessing_fingerprint(),
        "torch": str(torch.__version__),
        "checkpoint_code": sha256_file(Path(__file__)),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def save_replay_checkpoint(predictor, path, *, signature, row_offset):
    if predictor._last_key is None or predictor._last_key[0] != predictor._day:
        raise ValueError("a completed replay day is required")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=path.parent) as temporary:
        directory = Path(temporary) / "state"
        save_predictor(
            directory,
            predictor.models,
            predictor.features,
            predictor.scaler,
            predictor.config,
            predictor.seeds,
        )
        atomic_torch_save(
            [None if o is None else o.state_dict() for o in predictor._optimizers],
            directory / "optimizers.pt",
        )
        files = {"optimizers.pt": sha256_file(directory / "optimizers.pt")}
        if predictor._cache:
            public = pl.concat(predictor._cache)
            if any(c.startswith("responder_") for c in public.columns):
                raise ValueError("replay cache must contain only public inputs")
            public.write_parquet(directory / "public_cache.parquet")
            files["public_cache.parquet"] = sha256_file(directory / "public_cache.parquet")
        write_json(
            directory / "replay.json",
            {
                "signature": signature,
                "completed_date": predictor._day,
                "last_key": predictor._last_key,
                "row_offset": row_offset,
                "updates": predictor.update_log,
                "sha256": files,
                "resume_scope": "next date only; recurrent state resets at the day boundary",
            },
        )
        directory.rename(path)


def load_replay_checkpoint(path, *, signature, device):
    path = Path(path)
    state = json.loads((path / "replay.json").read_text())
    if state["signature"] != signature:
        raise ValueError("replay checkpoint identity mismatch")
    for name, digest in state["sha256"].items():
        if sha256_file(path / name) != digest:
            raise ValueError("replay checkpoint checksum mismatch")
    predictor = load_predictor(path, device)
    optimizer_states = torch.load(path / "optimizers.pt", map_location=device, weights_only=True)
    for index, (model, saved) in enumerate(zip(predictor.models, optimizer_states, strict=True)):
        if saved is not None:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=predictor.config.learning_rate, betas=predictor.config.betas
            )
            optimizer.load_state_dict(saved)
            predictor._optimizers[index] = optimizer
    predictor._day = state["completed_date"]
    predictor._last_key = tuple(state["last_key"])
    predictor._resume_after_day = state["completed_date"]
    predictor.update_log = state["updates"]
    if "public_cache.parquet" in state["sha256"]:
        predictor._cache = [pl.read_parquet(path / "public_cache.parquet")]
    return predictor, state
