"""Portable model weights, preprocessing state and run provenance."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.data.features import FeatureConfig, FeaturePipeline, TrainOnlyStandardizer
from src.models.grigoreva_gru import GrigorevaGRU, ModelConfig
from src.training.online import OnlineConfig, StreamingPredictor


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_predictor(path, models, features, scaler, online, seeds):
    """Save the initial inference state BEFORE validation updates occur."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    torch.save([m.state_dict() for m in models], path / "weights.pt")
    metadata = {
        "format_version": 1,
        "models": [asdict(m.config) for m in models],
        "features": asdict(features.config),
        "feature_names": features.names,
        "feature_state": features.state_dict(),
        "scaler": scaler.state_dict(),
        "online": asdict(online),
        "seeds": list(seeds),
        "weights_sha256": sha256_file(path / "weights.pt"),
    }
    write_json(path / "metadata.json", metadata)


def load_predictor(path, device="cpu", *, reset_clock=False):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata["format_version"] != 1
        or sha256_file(path / "weights.pt") != metadata["weights_sha256"]
    ):
        raise ValueError("unsupported or corrupt model artifact")
    features = FeaturePipeline(FeatureConfig(**metadata["features"]))
    if features.names != metadata["feature_names"]:
        raise ValueError("feature schema changed since artifact creation")
    features.load_state_dict(metadata["feature_state"])
    if reset_clock:
        features.reset_clock()
    scaler = TrainOnlyStandardizer.from_state_dict(metadata["scaler"])
    states = torch.load(path / "weights.pt", map_location=device, weights_only=True)
    models = []
    for config, state in zip(metadata["models"], states, strict=True):
        model = GrigorevaGRU(len(features.names), ModelConfig(**config)).to(device)
        model.load_state_dict(state)
        models.append(model.eval())
    return StreamingPredictor(
        models, features, scaler, OnlineConfig(**metadata["online"]), seeds=metadata["seeds"]
    )
