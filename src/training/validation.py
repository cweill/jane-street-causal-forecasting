"""Frozen epoch validation, with labels confined to the evaluator and separate cache."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.artifacts import sha256_file, write_json
from src.data.schema import KEYS, test_view
from src.metric import WeightedZeroMeanR2
from src.training.patrick import pack_panel


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PreparedValidation:
    directory: Path

    @property
    def manifest(self):
        return json.loads((self.directory / "manifest.json").read_text())

    def verify(self, training):
        manifest = self.manifest
        dates = tuple(manifest["dates"])
        if not dates or dates != tuple(sorted(set(dates))) or min(dates) <= max(training.dates):
            raise ValueError("validation dates must be after training")
        if (
            manifest["training_dates"] != list(training.dates)
            or manifest["features_sha256"] != fingerprint(training.features.state_dict())
            or sorted(training.scaler.training_dates) != list(training.dates)
        ):
            raise ValueError("validation preprocessing/training partition mismatch")
        if set(manifest["sha256"]) != {f"{d}.npz" for d in dates}:
            raise ValueError("validation cache coverage mismatch")
        for name, expected in manifest["sha256"].items():
            if sha256_file(self.directory / name) != expected:
                raise ValueError(f"validation checksum mismatch: {name}")
        return fingerprint(manifest)


def prepare_validation(source, dates, training, directory):
    dates = tuple(dates)
    if (
        not dates
        or dates != tuple(sorted(set(dates)))
        or min(dates) <= max(training.dates)
        or not set(dates).issubset(source.dates())
    ):
        raise ValueError("validation dates must be ordered and after training")
    if tuple(sorted(training.scaler.training_dates)) != tuple(training.dates):
        raise ValueError("preprocessing must be fitted on exactly the training partition")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for date in dates:
        day = source.day(date).sort(KEYS)
        # Responders never enter feature transformation or model inputs.
        public = test_view(day)
        arrays, keys = pack_panel(public, training.features)
        x, cats, mask = arrays[:3]
        indices = np.asarray([(t, s) for _, t, s in keys], dtype=np.int64)
        # Preserve source precision and original competition weights for scoring.
        np.savez(
            directory / f"{date}.npz",
            x=x,
            cats=cats,
            mask=mask,
            indices=indices,
            y=day["responder_6"].to_numpy().astype(np.float64),
            w=day["weight"].to_numpy().astype(np.float64),
            scored=public["is_scored"].to_numpy(),
        )
    write_json(
        directory / "manifest.json",
        {
            "format": 1,
            "dates": dates,
            "training_dates": training.dates,
            "features_sha256": fingerprint(training.features.state_dict()),
            "sha256": {f"{d}.npz": sha256_file(directory / f"{d}.npz") for d in dates},
        },
    )
    result = PreparedValidation(directory)
    result.verify(training)
    return result


def evaluate_validation(model, validation):
    """Whole-day causal forward, fresh state daily; no training or online updates.

    Existing prefix/streaming parity tests establish that the forward only sees
    the current cross-section and past observations. Labels are used exclusively
    by the float64 pooled metric, after predictions are produced.
    """
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    metric = WeightedZeroMeanR2()
    dates = validation.manifest["dates"]
    # A disposable copy protects modes, buffers, parameters and gradients. Forking
    # RNG also protects training if an inference implementation consumes randomness.
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        frozen = copy.deepcopy(model).eval().requires_grad_(False)
        for date in dates:
            with np.load(validation.directory / f"{date}.npz", allow_pickle=False) as data:
                prediction, _ = frozen(
                    *[torch.as_tensor(data[k], device=device) for k in ("x", "cats", "mask")]
                )
                indices = data["indices"]
                p = prediction[0, indices[:, 0], indices[:, 1], 6].cpu().numpy()
                metric.update(data["y"], p, data["w"], data["scored"])
    return {
        "r2": metric.score if metric.denominator > 0 else None,
        "sse": metric.sse,
        "denominator": metric.denominator,
        "rows": metric.rows,
        "date_start": dates[0],
        "date_end": dates[-1],
    }
