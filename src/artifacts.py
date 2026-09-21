"""Portable model weights, preprocessing state and run provenance."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


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


def save_predictor(path, models, features, scaler, online, seeds, *, stacked_inference=False):
    """Save the initial inference state BEFORE validation updates occur."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    torch.save([m.state_dict() for m in models], path / "weights.pt")
    from src.data.patrick_features import PatrickFeatures

    if not isinstance(features, PatrickFeatures):
        raise TypeError("Patrick preprocessing required")
    write_json(
        path / "metadata.json",
        {
            "format_version": 2,
            "method": "patrick",
            "models": [asdict(m.config) for m in models],
            "features": asdict(features.config),
            "feature_state": features.state_dict(),
            "online": asdict(online),
            "seeds": list(seeds),
            "stacked_inference": stacked_inference,
            "weights_sha256": sha256_file(path / "weights.pt"),
            "note": "Initial replay checkpoint; online optimizer/cache are not resumed.",
        },
    )


def load_predictor(path, device="cpu", *, reset_clock=False):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata["format_version"] != 2
        or metadata.get("method") != "patrick"
        or sha256_file(path / "weights.pt") != metadata["weights_sha256"]
    ):
        raise ValueError("unsupported or corrupt model artifact")
    from src.data.patrick_features import PatrickFeatureConfig, PatrickFeatures
    from src.models.patrick_yam import PatrickModelConfig, PatrickYam
    from src.training.patrick import PatrickOnlineConfig, PatrickPredictor

    features = PatrickFeatures(
        PatrickFeatureConfig(**metadata["features"]),
        metadata["feature_state"]["scaler"]["training_dates"],
    )
    features.load_state_dict(metadata["feature_state"])
    states = torch.load(path / "weights.pt", map_location=device, weights_only=True)
    models = []
    for config, state in zip(metadata["models"], states, strict=True):
        model = PatrickYam(PatrickModelConfig(**config), features.vocab_sizes).to(device)
        model.load_state_dict(state)
        models.append(model.eval())
    return PatrickPredictor(
        models,
        features,
        PatrickOnlineConfig(**metadata["online"]),
        metadata["seeds"],
        stacked_inference=metadata.get("stacked_inference", False),
    )
