"""Paired online hyperparameter comparisons on a predeclared development interval."""

import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from src.artifacts import sha256_file, write_json
from src.data.schema import validate_day
from src.parallel_replay import run_replay_mode
from src.plotting import rolling_r2


@dataclass(frozen=True)
class Trial:
    name: str
    learning_rate: float | None
    reset_daily: bool

    def __post_init__(self):
        if (
            not re.fullmatch(r"[a-z0-9_-]+", self.name)
            or type(self.reset_daily) is not bool
            or (
                self.learning_rate is not None
                and (not math.isfinite(self.learning_rate) or self.learning_rate <= 0)
            )
        ):
            raise ValueError("invalid trial configuration")
        if self.learning_rate is None and self.reset_daily:
            raise ValueError("frozen trial cannot reset optimizer")

    @property
    def mode(self):
        return "offline" if self.learning_rate is None else "online"


def trial_grid():
    return [Trial("frozen", None, False)] + [
        Trial(f"lr_{name}_{policy}", lr, reset)
        for name, lr in [("1e-4", 1e-4), ("5e-4", 5e-4), ("1e-3", 1e-3)]
        for policy, reset in [("persistent", False), ("reset", True)]
    ]


class ReplayDayCache:
    """Evaluator-owned daily parquet files; a predictor never receives this object."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self._dates = tuple(self.manifest["dates"])
        if (
            not self._dates
            or self._dates != tuple(sorted(set(self._dates)))
            or set(self.manifest["sha256"]) != {f"{d}.parquet" for d in self._dates}
        ):
            raise ValueError("invalid replay cache coverage")

    def dates(self):
        return self._dates

    def day(self, date):
        if date not in self._dates:
            raise ValueError("date outside replay partition")
        path = self.directory / f"{date}.parquet"
        if sha256_file(path) != self.manifest["sha256"][path.name]:
            raise ValueError("replay day checksum mismatch")
        day = pl.read_parquet(path)
        validate_day(day)
        if day["date_id"].unique().to_list() != [date]:
            raise ValueError("replay day identity mismatch")
        return day


def prepare_replay_days(source, dates, directory):
    dates, directory = tuple(dates), Path(directory)
    if not dates or dates != tuple(sorted(set(dates))) or not set(dates).issubset(source.dates()):
        raise ValueError("invalid replay partition")
    if directory.exists():
        cache = ReplayDayCache(directory)
        if cache.dates() != dates:
            raise ValueError("replay cache identity mismatch")
        for date in dates:
            if not cache.day(date).equals(source.day(date)):
                raise ValueError("replay cache differs from source")
        return cache
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".days-") as temporary:
        staged = Path(temporary) / "data"
        staged.mkdir()
        for date in dates:
            day = source.day(date)
            validate_day(day)
            if day["date_id"].unique().to_list() != [date]:
                raise ValueError("source date identity mismatch")
            day.write_parquet(staged / f"{date}.parquet")
        write_json(
            staged / "manifest.json",
            {
                "dates": dates,
                "sha256": {f"{d}.parquet": sha256_file(staged / f"{d}.parquet") for d in dates},
            },
        )
        staged.rename(directory)
    return ReplayDayCache(directory)


def validate_fold(fold):
    if not fold.train_dates or not fold.validation_dates or fold.validation_dates[-1] >= 1380:
        raise ValueError("development training/validation must end before 1380")
    all_dates = fold.train_dates + fold.replay_dates
    if all_dates != tuple(range(all_dates[0], all_dates[-1] + 1)):
        raise ValueError("consecutive disjoint training, warmup and validation dates required")


def trial_checkpoint(base, destination, trial, fold):
    base, destination = Path(base), Path(destination)
    validate_fold(fold)
    original = json.loads((base / "metadata.json").read_text())
    if original["feature_state"]["scaler"]["training_dates"] != list(fold.train_dates):
        raise ValueError("initial model training/preprocessing boundary mismatch")
    if sha256_file(base / "weights.pt") != original["weights_sha256"]:
        raise ValueError("initial checkpoint checksum mismatch")
    online = {**original["online"], "enabled": trial.learning_rate is not None}
    if trial.learning_rate is not None:
        online.update(learning_rate=trial.learning_rate, reset_daily_optimizer=trial.reset_daily)
    metadata = {**original, "online": online}
    if destination.exists():
        if (
            json.loads((destination / "metadata.json").read_text()) != metadata
            or sha256_file(destination / "weights.pt") != original["weights_sha256"]
        ):
            raise ValueError("trial checkpoint identity mismatch")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".initial-") as temporary:
        staged = Path(temporary) / "checkpoint"
        staged.mkdir()
        shutil.copyfile(base / "weights.pt", staged / "weights.pt")
        write_json(staged / "metadata.json", metadata)
        staged.rename(destination)


def run_trial(source, base, directory, trial, fold, *, device, provenance, progress=None):
    directory = Path(directory)
    trial_checkpoint(base, directory / "initial_checkpoint", trial, fold)
    result = run_replay_mode(
        source,
        directory / trial.mode,
        checkpoint=directory / "initial_checkpoint",
        mode=trial.mode,
        replay_dates=fold.replay_dates,
        scored_dates=fold.validation_dates,
        device=device,
        provenance={**provenance, "trial": asdict(trial)},
        fast=True,
        progress=progress,
    )
    result = {**result, "trial": asdict(trial)}
    write_json(directory / "result.json", result)
    return result


def summarize_trials(
    directory, trials, fold, window=20, *, trial_directories=None, expected_provenance=None
):
    validate_fold(fold)
    directory = Path(directory)
    if (
        len({t.name for t in trials}) != len(trials)
        or sum(t.learning_rate is None for t in trials) != 1
    ):
        raise ValueError("distinct trials and exactly one frozen baseline required")
    baseline = next(t for t in trials if t.learning_rate is None)
    roots = (
        {t.name: directory / t.name for t in trials}
        if trial_directories is None
        else {k: Path(v) for k, v in trial_directories.items()}
    )
    names = {t.name for t in trials}
    if set(roots) != names or (
        expected_provenance is not None and set(expected_provenance) != names
    ):
        raise ValueError("paired source/provenance coverage mismatch")
    frozen = json.loads((roots[baseline.name] / "result.json").read_text())
    frozen_identity = json.loads(
        (roots[baseline.name] / baseline.mode / "identity.json").read_text()
    )
    first = pl.read_parquet(
        roots[baseline.name] / baseline.mode / f"date_{fold.replay_dates[0]}.parquet"
    )
    rows, curves = [], []
    for trial in trials:
        root = roots[trial.name]
        result = json.loads((root / "result.json").read_text())
        identity = json.loads((root / trial.mode / "identity.json").read_text())
        if (
            result["trial"] != asdict(trial)
            or identity["replay_dates"] != list(fold.replay_dates)
            or identity["scored_dates"] != list(fold.validation_dates)
            or identity["checkpoint_weights_sha256"] != frozen_identity["checkpoint_weights_sha256"]
            or {k: v for k, v in identity["provenance"].items() if k != "trial"}
            != (
                expected_provenance[trial.name]
                if expected_provenance is not None
                else {k: v for k, v in frozen_identity["provenance"].items() if k != "trial"}
            )
            or any(
                result[k] != frozen[k]
                for k in ("initial_weights_sha256", "denominator", "scored_rows")
            )
            or not pl.read_parquet(
                root / trial.mode / f"date_{fold.replay_dates[0]}.parquet"
            ).equals(first)
        ):
            raise ValueError("paired trial identities/rows/initial predictions differ")
        rows.append(
            {
                **asdict(trial),
                "r2": result["score"],
                "delta_r2": result["score"] - frozen["score"],
                "seconds": result["seconds"],
            }
        )
        curves.append(
            rolling_r2(pl.read_parquet(root / trial.mode / "daily.parquet"), window).with_columns(
                pl.lit(trial.name).alias("trial")
            )
        )
    table = pl.concat(curves)
    directory.mkdir(parents=True, exist_ok=True)
    table.write_csv(directory / "rolling_comparison.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for trial, curve in zip(trials, curves, strict=True):
        ax.plot(
            curve["date_id"],
            curve["r2"],
            label=trial.name,
            linestyle="--" if trial.reset_daily else "-",
            linewidth=2 if trial.learning_rate is None else 1.1,
            **({"color": "black"} if trial.learning_rate is None else {}),
        )
    ax.axvline(fold.validation_dates[0], color="gray", linestyle=":", label="Scoring starts")
    ax.set(
        xlabel="date_id",
        ylabel=f"Rolling {window}-day weighted zero-mean R²",
        title="Online learning sensitivity · fixed development checkpoint",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "rolling_comparison.png", dpi=160)
    plt.close(fig)
    summary = {
        "status": "complete",
        "trials": rows,
        "scored_dates": list(fold.validation_dates),
        "note": "Development sensitivity study; all candidates reported, no automatic promotion.",
    }
    write_json(directory / "result.json", summary)
    return summary
