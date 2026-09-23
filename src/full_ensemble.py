"""Controls for a staged nine-epoch ensemble and its fixed five-epoch baseline."""

import json
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import torch

from src.artifacts import sha256_file, write_json
from src.cv import configured_folds
from src.plotting import rolling_r2


def read(path):
    return json.loads(Path(path).read_text())


def full_fold(config, reference, dates):
    expected = replace(
        reference,
        name="patrick_nine_epoch_ensemble",
        training=replace(reference.training, epochs=9),
        online=replace(reference.online, learning_rate=1e-4),
    )
    if config != expected or config.ensemble.seeds != tuple(range(17)):
        raise ValueError("full ensemble protocol allows only nine epochs and online LR 1e-4")
    fold = configured_folds(dates, config.cv)[0]
    if (
        fold.train_dates != tuple(range(1380))
        or fold.warmup_dates != tuple(range(1380, 1500))
        or fold.validation_dates != tuple(range(1500, 1699))
    ):
        raise ValueError("full ensemble protocol boundaries differ")
    return fold


def require_seed_stage(seed, pilot_gate, code_sha):
    if type(seed) is not int or seed not in range(17):
        raise ValueError("invalid ensemble seed")
    if seed >= 3 and (
        not pilot_gate
        or pilot_gate.get("passed") is not True
        or pilot_gate.get("code_sha256") != code_sha
    ):
        raise ValueError("matching completed pilot required before remaining seeds")


def check_epoch_five(saved, reference):
    if saved["epoch"] != 5 or saved["position"] != 0:
        return False
    if saved["model"].keys() != reference.keys() or any(
        not torch.equal(v, reference[k]) for k, v in saved["model"].items()
    ):
        raise ValueError("epoch-five control weights differ from original")
    return True


def checked_daily(path, dates, scored):
    path = Path(path)
    result, identity = read(path / "result.json"), read(path / "identity.json")
    daily = pl.read_parquet(path / "daily.parquet")
    if (
        identity["replay_dates"] != list(dates)
        or identity["scored_dates"] != list(scored)
        or daily["date_id"].to_list() != list(dates)
    ):
        raise ValueError("replay coverage dates differ")
    # Also validates finite, nonnegative sufficient statistics.
    rolling_r2(daily, 1)
    selected = daily.filter(pl.col("date_id").is_in(scored))
    sse, energy = selected["sse"].sum(), selected["denominator"].sum()
    if (
        selected["rows"].sum() != result["scored_rows"]
        or not math.isclose(sse, result["sse"], rel_tol=1e-12, abs_tol=1e-10)
        or not math.isclose(energy, result["denominator"], rel_tol=1e-12, abs_tol=1e-10)
        or not math.isclose(1 - sse / energy, result["score"], rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("replay coverage or pooled score differs from daily statistics")
    return result, identity, daily


def verify_pair(root, dates, scored, seeds):
    root = Path(root)
    metadata = read(root / "initial_checkpoint/metadata.json")
    if (
        metadata["seeds"] != list(seeds)
        or metadata["feature_state"]["scaler"]["training_dates"][-1] != dates[0] - 1
        or metadata["online"]["learning_rate"] != 1e-4
        or metadata["online"]["reset_daily_optimizer"]
        or metadata["online"]["steps"] != 3
    ):
        raise ValueError("paired checkpoint seed/training/online configuration mismatch")
    results = {}
    reference_identity, reference_daily = None, None
    for mode in ("offline", "online"):
        result, identity, daily = checked_daily(root / mode, dates, scored)
        if (
            identity["mode"] != mode
            or identity["checkpoint_weights_sha256"] != metadata["weights_sha256"]
            or identity["checkpoint_metadata_sha256"]
            != sha256_file(root / "initial_checkpoint/metadata.json")
            or sha256_file(root / "initial_checkpoint/weights.pt") != metadata["weights_sha256"]
        ):
            raise ValueError("paired checkpoint identity mismatch")
        if reference_identity is not None and (
            identity["provenance"] != reference_identity["provenance"]
            or not daily.select("date_id", "rows", "denominator").equals(
                reference_daily.select("date_id", "rows", "denominator")
            )
        ):
            raise ValueError("paired coverage/provenance mismatch")
        reference_identity, reference_daily = identity, daily
        results[mode] = result
    a, b = results["offline"], results["online"]
    if any(a[k] != b[k] for k in ("initial_weights_sha256", "denominator", "scored_rows")):
        raise ValueError("paired initial weights/coverage mismatch")
    if a["initial_weights_sha256"] != a["final_weights_sha256"] or a["updates"]:
        raise ValueError("frozen weights changed")
    updates = b["updates"]
    if len(updates) != len(dates) - 1 or any(
        u["source_date"] != d - 1
        or u["released_at"] != [d, 0]
        or u["models"] != len(seeds)
        or u["steps"] != 3
        or u["optimizer_reset_daily"]
        or u["target"] != "all_9"
        for d, u in zip(dates[1:], updates, strict=True)
    ):
        raise ValueError("online update count, timing or recipe mismatch")
    first = f"date_{dates[0]}.parquet"
    if not pl.read_parquet(root / "offline" / first).equals(
        pl.read_parquet(root / "online" / first)
    ):
        raise ValueError("first-day predictions differ before label release")
    return {
        "passed": True,
        "frozen_r2": a["score"],
        "online_r2": b["score"],
        "updates": len(updates),
        "scored_rows": a["scored_rows"],
        "denominator": a["denominator"],
        "initial_weights_sha256": a["initial_weights_sha256"],
    }


def compare_baseline(root, baseline_paths, fold, destination, *, seeds=tuple(range(17)), window=20):
    paired = verify_pair(root, fold.replay_dates, fold.validation_dates, seeds)
    paths = {
        "five_frozen": Path(baseline_paths["offline"]),
        "five_online": Path(baseline_paths["online"]),
        "nine_frozen": Path(root) / "offline",
        "nine_online": Path(root) / "online",
    }
    results, identities, curves, blocks = {}, {}, [], []
    reference = None
    for label, path in paths.items():
        result, identity, daily = checked_daily(path, fold.replay_dates, fold.validation_dates)
        if reference is not None and not daily.select("date_id", "rows", "denominator").equals(
            reference
        ):
            raise ValueError("baseline coverage differs from nine-epoch evaluation")
        reference = daily.select("date_id", "rows", "denominator")
        results[label], identities[label] = result, identity
        curves.append(rolling_r2(daily, window).with_columns(pl.lit(label).alias("configuration")))
        for offset in range(0, len(fold.validation_dates), window):
            dates = fold.validation_dates[offset : offset + window]
            selected = daily.filter(pl.col("date_id").is_in(dates))
            blocks.append(
                {
                    "configuration": label,
                    "start_date": dates[0],
                    "end_date": dates[-1],
                    "days": len(dates),
                    "r2": 1 - selected["sse"].sum() / selected["denominator"].sum(),
                }
            )
    if (
        results["five_frozen"]["initial_weights_sha256"]
        != results["five_online"]["initial_weights_sha256"]
    ):
        raise ValueError("baseline weights differ between frozen and online")
    if (
        identities["five_frozen"]["checkpoint_weights_sha256"]
        != identities["five_online"]["checkpoint_weights_sha256"]
    ):
        raise ValueError("baseline checkpoint mismatch")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pl.concat(curves).write_csv(destination / "rolling_comparison.csv")
    pl.DataFrame(blocks).write_csv(destination / "scored_blocks.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for label, curve in zip(paths, curves, strict=True):
        ax.plot(
            curve["date_id"],
            curve["r2"],
            label=label,
            linestyle="--" if label.endswith("frozen") else "-",
            color="C0" if label.startswith("five") else "C1",
        )
    ax.axvline(fold.validation_dates[0], color="gray", linestyle=":", label="Scoring starts")
    ax.set(
        xlabel="date_id",
        ylabel=f"Rolling {window}-day weighted zero-mean R²",
        title="17-model ensemble · five versus nine offline epochs",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(destination / "comparison.png", dpi=160)
    plt.close(fig)
    summary = {
        "status": "complete",
        **{k: v["score"] for k, v in results.items()},
        "online_delta": results["nine_online"]["score"] - results["five_online"]["score"],
        "frozen_delta": results["nine_frozen"]["score"] - results["five_frozen"]["score"],
        "paired_verification": paired,
        "note": "Previously inspected later-date evaluation; not an untouched holdout.",
    }
    write_json(destination / "result.json", summary)
    return summary


def replay_metrics(records, mode, *, window=20):
    """Build live metrics from the committed prefix retained by one replay worker."""
    latest = records[-1]
    primary_energy = sum(r["primary"]["denominator"] for r in records)
    metrics = {
        f"{mode}/date_id": latest["date_id"],
        f"{mode}/days_completed": len(records),
        f"{mode}/day_seconds": latest["seconds"],
        f"{mode}/is_scored": bool(latest["primary"]["rows"]),
    }
    if latest["diagnostic"]["denominator"] > 0:
        metrics[f"{mode}/diagnostic_daily_r2"] = (
            1 - latest["diagnostic"]["sse"] / latest["diagnostic"]["denominator"]
        )
    if primary_energy > 0:
        metrics[f"{mode}/scored_cumulative_r2"] = (
            1 - sum(r["primary"]["sse"] for r in records) / primary_energy
        )
    if len(records) >= window:
        tail = records[-window:]
        energy = sum(r["diagnostic"]["denominator"] for r in tail)
        if energy > 0:
            metrics[f"{mode}/rolling_r2"] = 1 - sum(r["diagnostic"]["sse"] for r in tail) / energy
    return metrics
