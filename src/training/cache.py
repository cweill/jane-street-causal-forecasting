"""Reusable, checksummed Patrick preprocessing; model/optimizer settings are not cache keys."""

import hashlib
import importlib.metadata
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.artifacts import sha256_file, write_json
from src.data.patrick_features import PatrickFeatures
from src.training.patrick import PreparedPatrick, prepare_training

ROOT = Path(__file__).resolve().parents[2]


def parquet_fingerprint(path):
    """Content identity is independent of the local/Modal parent directory."""
    path = Path(path)
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError(path)
    files = [
        {
            "name": str(p.relative_to(path)) if path.is_dir() else p.name,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in paths
    ]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


def preprocessing_fingerprint():
    digest = hashlib.sha256()
    for name in (
        "src/data/schema.py",
        "src/data/normalization.py",
        "src/data/patrick_features.py",
        "src/training/patrick.py",
        "src/training/cache.py",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def prepare_cached(source, dates, feature_config, cache_root, dataset_sha256):
    """dataset_sha256 must identify immutable, checksum-verified source contents.

    Callers must verify this identity against their files before calling this helper.
    A cache hit validates every prepared file and never reads raw feature/target rows.
    """
    dates = tuple(dates)
    if not dates or dates != tuple(sorted(set(dates))) or not set(dates).issubset(source.dates()):
        raise ValueError("invalid cached training dates")
    if len(dataset_sha256) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha256):
        raise ValueError("verified SHA256 dataset identity required")
    identity = json.loads(
        json.dumps(
            {
                "format": 1,
                "dataset_sha256": dataset_sha256,
                "dates": dates,
                "features": asdict(feature_config),
                "preprocessing_sha256": preprocessing_fingerprint(),
                "versions": {
                    name: importlib.metadata.version(name) for name in ("numpy", "polars")
                },
            }
        )
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    directory = cache_root / key
    hit = directory.exists()
    if not hit:
        # Partial builds never become cache hits. Rename only after all files and
        # the completion manifest are written; do not overwrite another writer.
        with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as temp:
            prepared = prepare_training(source, dates, feature_config, Path(temp) / "data")
            write_json(prepared.directory / "features.json", prepared.features.state_dict())
            files = [*(f"{date}.npz" for date in dates), "features.json"]
            write_json(
                prepared.directory / "manifest.json",
                {
                    "identity": identity,
                    "sha256": {name: sha256_file(prepared.directory / name) for name in files},
                },
            )
            try:
                prepared.directory.rename(directory)
            except OSError:
                if not directory.exists():
                    raise
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {*(f"{date}.npz" for date in dates), "features.json"}
    if manifest["identity"] != identity or set(manifest["sha256"]) != expected:
        raise ValueError("cache manifest identity/coverage mismatch")
    for name, digest in manifest["sha256"].items():
        if not (directory / name).is_file() or sha256_file(directory / name) != digest:
            raise ValueError(f"cache checksum mismatch: {name}")
    state = json.loads((directory / "features.json").read_text())
    if state["scaler"]["training_dates"] != list(dates):
        raise ValueError("cache preprocessing has wrong training dates")
    features = PatrickFeatures(feature_config, dates)
    features.load_state_dict(state)
    return PreparedPatrick(directory, dates, features), {
        "key": key,
        "hit": hit,
        "directory": str(directory),
    }
