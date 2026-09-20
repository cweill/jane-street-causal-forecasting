"""Reuse a completed ensemble for a single, paired lower-LR online replay."""

import json
import shutil
import tempfile
from pathlib import Path

import polars as pl

from src.artifacts import sha256_file, write_json
from src.plotting import rolling_r2


def read(path):
    return json.loads(Path(path).read_text())


def prepare_followup(original, root, learning_rate=1e-4):
    original, root = Path(original), Path(root)
    if learning_rate != 1e-4:
        raise ValueError("follow-up identity fixes learning rate at 1e-4")
    base = original / "initial_checkpoint"
    metadata = read(base / "metadata.json")
    identities = {mode: read(original / mode / "identity.json") for mode in ("offline", "online")}
    a, b = identities["offline"], identities["online"]
    for key in (
        "checkpoint_weights_sha256",
        "checkpoint_metadata_sha256",
        "replay_dates",
        "scored_dates",
        "provenance",
        "fast",
    ):
        if a[key] != b[key]:
            raise ValueError("reference replay identity mismatch")
    if (
        sha256_file(base / "weights.pt") != a["checkpoint_weights_sha256"]
        or metadata["weights_sha256"] != a["checkpoint_weights_sha256"]
        or sha256_file(base / "metadata.json") != a["checkpoint_metadata_sha256"]
    ):
        raise ValueError("reference initial checkpoint mismatch")
    dates, scored = a["replay_dates"], a["scored_dates"]
    trained = metadata["feature_state"]["scaler"]["training_dates"]
    if (
        not dates
        or dates != list(range(dates[0], dates[-1] + 1))
        or not scored
        or scored != list(range(scored[0], dates[-1] + 1))
        or scored[0] < dates[0]
        or not trained
        or trained != list(range(trained[0], dates[0]))
    ):
        raise ValueError("reference training/replay boundary mismatch")
    if (
        not metadata["online"]["enabled"]
        or metadata["online"]["reset_daily_optimizer"]
        or metadata["online"]["learning_rate"] != 5e-4
        or not metadata["stacked_inference"]
    ):
        raise ValueError("reference must use stacked inference and persistent Adam at 5e-4")
    record = {
        "learning_rate": learning_rate,
        "replay_dates": dates,
        "scored_dates": scored,
        "reference_identities": identities,
        "initial_metadata_sha256": a["checkpoint_metadata_sha256"],
        "initial_weights_sha256": a["checkpoint_weights_sha256"],
    }
    root.mkdir(parents=True, exist_ok=True)
    target = root / "initial_checkpoint"
    metadata["online"]["learning_rate"] = learning_rate
    if target.exists():
        if (
            read(target / "metadata.json") != metadata
            or sha256_file(target / "weights.pt") != a["checkpoint_weights_sha256"]
        ):
            raise ValueError("follow-up checkpoint identity mismatch")
    else:
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            staged = Path(temporary) / "checkpoint"
            staged.mkdir()
            shutil.copyfile(base / "weights.pt", staged / "weights.pt")
            write_json(staged / "metadata.json", metadata)
            staged.rename(target)
    for mode in ("offline", "online"):
        destination = root / "reference" / mode
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("identity.json", "result.json", "daily.parquet", f"date_{dates[0]}.parquet"):
            source, dest = original / mode / name, destination / name
            if dest.exists() and sha256_file(source) != sha256_file(dest):
                raise ValueError("reference artifact changed")
            if not dest.exists():
                shutil.copyfile(source, dest)
    if (root / "followup.json").exists() and read(root / "followup.json") != record:
        raise ValueError("follow-up identity mismatch")
    write_json(root / "followup.json", record)
    return record


def check_first_day(root):
    root = Path(root)
    date = read(root / "followup.json")["replay_dates"][0]
    actual = pl.read_parquet(root / "online" / f"date_{date}.parquet")
    for mode in ("offline", "online"):
        if not actual.equals(pl.read_parquet(root / "reference" / mode / f"date_{date}.parquet")):
            raise ValueError("first-day predictions differ before any online updates")


def finish_followup(root, window=20):
    root = Path(root)
    record = read(root / "followup.json")
    check_first_day(root)
    results = {
        "frozen": read(root / "reference/offline/result.json"),
        "online_5e-4": read(root / "reference/online/result.json"),
        "online_1e-4": read(root / "online/result.json"),
    }
    identity = read(root / "online/identity.json")
    if (
        any(identity[k] != record[k] for k in ("replay_dates", "scored_dates"))
        or identity["checkpoint_weights_sha256"] != record["initial_weights_sha256"]
    ):
        raise ValueError("follow-up replay identity mismatch")
    for result in results.values():
        if any(
            result[k] != results["frozen"][k]
            for k in ("initial_weights_sha256", "scored_rows", "denominator")
        ):
            raise ValueError("paired score coverage/initial weights mismatch")
    curves = []
    for name, path in [
        ("frozen", "reference/offline"),
        ("online_5e-4", "reference/online"),
        ("online_1e-4", "online"),
    ]:
        daily = pl.read_parquet(root / path / "daily.parquet")
        if daily["date_id"].to_list() != record["replay_dates"]:
            raise ValueError("paired daily coverage mismatch")
        curves.append(rolling_r2(daily, window).with_columns(pl.lit(name).alias("mode")))
    table = pl.concat(curves)
    table.write_csv(root / "comparison.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for name, curve in zip(results, curves, strict=True):
        ax.plot(curve["date_id"], curve["r2"], label=name)
    ax.axvline(record["scored_dates"][0], color="gray", linestyle=":", label="Scoring starts")
    ax.set(
        xlabel="date_id",
        ylabel=f"Rolling {window}-day weighted zero-mean R²",
        title="Fixed ensemble · online learning-rate follow-up",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    summary = {
        "status": "complete",
        **results,
        "delta_from_frozen": results["online_1e-4"]["score"] - results["frozen"]["score"],
        "delta_from_previous_online": results["online_1e-4"]["score"]
        - results["online_5e-4"]["score"],
        "note": "Follow-up on a previously inspected interval; not an untouched test.",
    }
    write_json(root / "result.json", summary)
    return summary
