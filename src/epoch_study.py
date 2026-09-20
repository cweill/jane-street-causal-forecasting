"""Immutable epoch artifacts and paired frozen/online epoch-budget comparisons."""

import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import polars as pl
import torch

from src.artifacts import load_predictor, save_predictor, write_json
from src.ensemble_run import config_fingerprint
from src.models.patrick_yam import PatrickYam
from src.online_refinement import block_comparison
from src.online_sweep import Trial, validate_fold
from src.parallel_replay import ensemble_fingerprint
from src.plotting import rolling_r2


def read(path):
    return json.loads(Path(path).read_text())


def check_history(actual, reference, atol=1e-7):
    """Compare the deterministic prefix, allowing only small floating-point differences."""

    def compare(a, b):
        if isinstance(a, dict):
            return (
                isinstance(b, dict)
                and a.keys() == b.keys()
                and all(compare(v, b[k]) for k, v in a.items())
            )
        if isinstance(a, list):
            return (
                isinstance(b, list)
                and len(a) == len(b)
                and all(compare(x, y) for x, y in zip(a, b, strict=True))
            )
        if isinstance(a, float):
            return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-7, abs_tol=atol)
        return a == b

    if len(actual) > len(reference) or not compare(actual, reference[: len(actual)]):
        raise ValueError("new training history differs from original deterministic prefix")


def snapshot_epoch(saved, prepared, config, seed, directory):
    epoch = saved["epoch"]
    if saved["position"] != 0 or epoch not in (3, 4):
        return False
    if (
        config.training.epochs != 4
        or config.online.learning_rate != 1e-4
        or config.online.reset_daily_optimizer
        or saved["completed"] != epoch * len(prepared.dates)
        or len(saved["history"]) != epoch
        or seed not in config.ensemble.seeds
    ):
        raise ValueError("epoch artifact identity mismatch")
    directory = Path(directory)
    destination = directory / f"epoch_{epoch}"
    epoch_config = replace(config, training=replace(config.training, epochs=epoch))
    record = {
        "status": "complete",
        "seed": seed,
        "epochs": epoch,
        "config_sha256": config_fingerprint(epoch_config),
        "source_training_signature": saved["signature"],
        "history": saved["history"],
    }
    # Reconstructing CPU models must not consume the live training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = PatrickYam(config.model, prepared.features.vocab_sizes)
        model.load_state_dict(saved["model"])
        model.eval()
        if destination.exists():
            existing = load_predictor(destination / "checkpoint")
            if (
                read(destination / "result.json") != record
                or existing.features.state_dict() != prepared.features.state_dict()
                or ensemble_fingerprint(existing.models) != ensemble_fingerprint([model])
            ):
                raise ValueError("immutable epoch artifact identity mismatch")
            return True
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory, prefix=".epoch-") as temporary:
            staged = Path(temporary) / "artifact"
            staged.mkdir()
            save_predictor(
                staged / "checkpoint",
                [model],
                prepared.features,
                prepared.scaler,
                config.online,
                [seed],
            )
            write_json(staged / "result.json", record)
            staged.rename(destination)
    return True


def compare_epochs(roots, fold, directory, window=20):
    validate_fold(fold)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, curves, blocks = [], [], []
    reference = None
    dataset = None
    for epoch, paths in sorted(roots.items()):
        if set(paths) != {"offline", "online"}:
            raise ValueError("paired frozen and online paths required")
        paths = {k: Path(v) for k, v in paths.items()}
        results = {k: read(p / "result.json") for k, p in paths.items()}
        identities = {k: read(p / "identity.json") for k, p in paths.items()}
        a, b = results["offline"], results["online"]
        if any(a[k] != b[k] for k in ("initial_weights_sha256", "denominator", "scored_rows")):
            raise ValueError("paired epoch weights/coverage mismatch")
        for identity in identities.values():
            if identity["replay_dates"] != list(fold.replay_dates) or identity[
                "scored_dates"
            ] != list(fold.validation_dates):
                raise ValueError("paired epoch dates mismatch")
            if dataset is None:
                dataset = identity["provenance"]["dataset_sha256"]
            if identity["provenance"]["dataset_sha256"] != dataset:
                raise ValueError("paired epoch dataset mismatch")
        if (
            identities["offline"]["checkpoint_weights_sha256"]
            != identities["online"]["checkpoint_weights_sha256"]
        ):
            raise ValueError("paired checkpoint weights mismatch")
        first = f"date_{fold.replay_dates[0]}.parquet"
        if not pl.read_parquet(paths["offline"] / first).equals(
            pl.read_parquet(paths["online"] / first)
        ):
            raise ValueError("paired first-day predictions mismatch")
        if reference is None:
            reference = a
        if any(a[k] != reference[k] for k in ("denominator", "scored_rows")):
            raise ValueError("paired epoch scoring coverage mismatch")
        if a["initial_weights_sha256"] != a["final_weights_sha256"]:
            raise ValueError("frozen epoch weights changed")
        rows.append(
            {
                "epoch": epoch,
                "frozen_r2": a["score"],
                "online_r2": b["score"],
                "delta_r2": b["score"] - a["score"],
                "scored_rows": a["scored_rows"],
                "denominator": a["denominator"],
                "initial_weights_sha256": a["initial_weights_sha256"],
                "identities": identities,
            }
        )
        for mode, path in paths.items():
            day = pl.read_parquet(path / "daily.parquet")
            if day["date_id"].to_list() != list(fold.replay_dates):
                raise ValueError("paired daily coverage mismatch")
            curves.append(
                rolling_r2(day, window).with_columns(
                    pl.lit(epoch).alias("epoch"), pl.lit(mode).alias("mode")
                )
            )
        trials = [Trial("frozen", None, False), Trial("online", 1e-4, False)]
        blocks.append(
            block_comparison(
                {"frozen": paths["offline"].parent, "online": paths["online"].parent},
                trials,
                fold,
                block_days=window,
            ).with_columns(pl.lit(epoch).alias("epoch"))
        )
    pl.concat(curves).write_csv(directory / "rolling_comparison.csv")
    pl.concat(blocks).write_csv(directory / "scored_blocks.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    colors = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c"}
    for curve in curves:
        epoch = int(curve["epoch"][0])
        mode = curve["mode"][0]
        ax.plot(
            curve["date_id"],
            curve["r2"],
            color=colors[epoch],
            linestyle="-" if mode == "online" else "--",
            label=f"Epoch {epoch} · " + ("OL 1e-4" if mode == "online" else "frozen"),
        )
    ax.axvline(fold.validation_dates[0], color="gray", linestyle=":", label="Scoring starts")
    ax.set(
        xlabel="date_id",
        ylabel=f"Rolling {window}-day weighted zero-mean R²",
        title="Offline epoch budget · fixed three-seed ensemble",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "rolling_comparison.png", dpi=160)
    plt.close(fig)
    result = {
        "status": "complete",
        "epochs": rows,
        "scored_dates": list(fold.validation_dates),
        "note": "Development epoch comparison; no automatic selection or final-ensemble retraining.",
    }
    write_json(directory / "result.json", result)
    return result
