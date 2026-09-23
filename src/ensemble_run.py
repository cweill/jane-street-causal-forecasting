"""Verified cache reuse and atomic assembly of independently trained Patrick seeds."""

import ast
import hashlib
import importlib.metadata
import json
import tarfile
import tempfile
from pathlib import Path

from src.artifacts import load_predictor, save_predictor, sha256_file
from src.data.patrick_features import PatrickFeatures
from src.training.cache import ROOT
from src.training.patrick import PreparedPatrick


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()


def load_verified_cache(directory, source_archive, config, dates, dataset_sha256):
    """Accept a legacy cache only after proving its preparation code is unchanged.

    The old key hashes the whole training module, including unrelated inference
    code. Compare the actual preparation definitions, full feature modules, pinned
    numeric libraries, configuration, dates, and every prepared file instead.
    """
    directory = Path(directory)
    names = (
        "src/data/schema.py",
        "src/data/normalization.py",
        "src/data/patrick_features.py",
        "src/training/patrick.py",
        "src/training/cache.py",
    )
    old = {}
    with tarfile.open(source_archive, "r:gz") as archive:
        for name in names:
            member = archive.extractfile(name)
            if member is None:
                raise ValueError("missing preparation source")
            old[name] = member.read()
    for name in names[:3]:
        if old[name] != (ROOT / name).read_bytes():
            raise ValueError("feature preparation source changed")

    def definitions(data):
        wanted = {"PreparedPatrick", "pack_panel", "prepare_training"}
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(data).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
        }

    if definitions(old[names[3]]) != definitions((ROOT / names[3]).read_bytes()):
        raise ValueError("training preparation source changed")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(old[name])
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = manifest["identity"]
    expected = {
        "format": 1,
        "dataset_sha256": dataset_sha256,
        "dates": list(dates),
        "features": config.to_dict()["features"],
        "preprocessing_sha256": digest.hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "polars")},
    }
    if identity != canonical(expected):
        raise ValueError("cache identity mismatch")
    files = {"features.json", *(f"{d}.npz" for d in dates)}
    if set(manifest["sha256"]) != files:
        raise ValueError("cache date coverage mismatch")
    for name, expected_sha in manifest["sha256"].items():
        if sha256_file(directory / name) != expected_sha:
            raise ValueError(f"cache checksum mismatch: {name}")
    state = json.loads((directory / "features.json").read_text())
    if state["scaler"]["training_dates"] != list(dates):
        raise ValueError("cache scaler training boundary mismatch")
    features = PatrickFeatures(config.features, tuple(dates))
    features.load_state_dict(state)
    return PreparedPatrick(directory, tuple(dates), features)


def assemble_ensemble(seed_directories, config, destination, *, seeds=None):
    configured = tuple(seed for _, seed in config.members)
    seeds = configured if seeds is None else tuple(seeds)
    if (
        not seeds
        or not set(seeds).issubset(configured)
        or len(seed_directories) != len(seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("one completed artifact per distinct seed required")
    models, features = [], None
    for seed, directory in zip(seeds, seed_directories, strict=True):
        directory = Path(directory)
        record = json.loads((directory / "result.json").read_text())
        if record["seed"] != seed:
            raise ValueError("seed identity mismatch")
        if (
            record["status"] != "complete"
            or record["epochs"] != config.training.epochs
            or record["config_sha256"] != config_fingerprint(config)
        ):
            raise ValueError("seed is not complete for the requested configuration")
        predictor = load_predictor(directory / "checkpoint")
        if predictor.seeds != (seed,) or len(predictor.models) != 1:
            raise ValueError("artifact seed identity mismatch")
        if canonical(predictor.models[0].config.__dict__) != canonical(config.to_dict()["model"]):
            raise ValueError("member architecture mismatch")
        if canonical(predictor.features.config.__dict__) != canonical(config.to_dict()["features"]):
            raise ValueError("member preprocessing configuration mismatch")
        if features is None:
            features = predictor.features
        elif features.state_dict() != predictor.features.state_dict():
            raise ValueError("member preprocessing mismatch")
        models.append(predictor.models[0])
    destination = Path(destination)
    if destination.exists():
        from src.parallel_replay import ensemble_fingerprint

        existing = load_predictor(destination)
        if (
            existing.seeds != seeds
            or not existing.stacked_inference
            or existing.features.state_dict() != features.state_dict()
            or ensemble_fingerprint(existing.models) != ensemble_fingerprint(models)
        ):
            raise ValueError("existing ensemble differs from completed seeds")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ensemble-", dir=destination.parent) as temp:
        staged = Path(temp) / "checkpoint"
        save_predictor(
            staged, models, features, features.scaler, config.online, seeds, stacked_inference=True
        )
        staged.rename(destination)
