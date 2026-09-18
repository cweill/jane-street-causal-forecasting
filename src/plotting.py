"""Diagnostic rolling zero-mean R² from evaluator-owned daily sufficient statistics."""

from pathlib import Path

import numpy as np
import polars as pl

from src.data.schema import KEYS
from src.metric import WeightedZeroMeanR2


def score_day(predictions, truth):
    """Score all observed rows, including unscored warmup, for the diagnostic plot."""
    if truth["date_id"].n_unique() != 1:
        raise ValueError("one truth day required")
    joined = predictions.select(*KEYS, pl.col("responder_6").alias("prediction")).join(
        truth.select(*KEYS, "weight", "responder_6"), on=KEYS, how="inner", validate="1:1"
    )
    if joined.height != predictions.height or joined.height != truth.height:
        raise ValueError("prediction/truth coverage mismatch")
    metric = WeightedZeroMeanR2()
    metric.update(joined["responder_6"], joined["prediction"], joined["weight"])
    return {
        "date_id": int(truth["date_id"][0]),
        "rows": metric.rows,
        "sse": metric.sse,
        "denominator": metric.denominator,
    }


def rolling_r2(daily, window=20):
    if type(window) is not int or window < 1:
        raise ValueError("positive integer window required")
    dates = daily["date_id"].to_numpy()
    if not len(dates) or np.any(np.diff(dates) != 1):
        raise ValueError("daily statistics must have consecutive ascending dates")
    sse, energy = [daily[c].to_numpy().astype(np.float64) for c in ("sse", "denominator")]
    if (
        not np.isfinite(sse).all()
        or not np.isfinite(energy).all()
        or (sse < 0).any()
        or (energy < 0).any()
    ):
        raise ValueError("invalid daily sufficient statistics")
    rows = []
    for end in range(window - 1, len(dates)):
        start = end - window + 1
        denominator = energy[start : end + 1].sum(dtype=np.float64)
        if denominator <= 0:
            raise ValueError("rolling R² is undefined for zero target energy")
        rows.append(
            {
                "date_id": int(dates[end]),
                "window_start": int(dates[start]),
                "r2": 1 - sse[start : end + 1].sum(dtype=np.float64) / denominator,
            }
        )
    return pl.DataFrame(
        rows, schema={"date_id": pl.Int64, "window_start": pl.Int64, "r2": pl.Float64}
    )


def plot_online_comparison(offline, online, output, *, title, window=20, scored_start=None):
    if offline["date_id"].to_list() != online["date_id"].to_list():
        raise ValueError("both replays must cover the same dates")
    a, b = rolling_r2(offline, window), rolling_r2(online, window)
    if a.is_empty():
        raise ValueError("not enough dates for a complete rolling window")
    start = int(offline["date_id"][0])
    table = a.rename({"r2": "without_online"}).with_columns(
        b["r2"].alias("with_online"), (pl.col("date_id") - start).alias("day_offset")
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {kind: str(output.with_suffix("." + kind)) for kind in ("png", "svg", "csv")}
    if any(Path(path).exists() for path in paths.values()):
        raise FileExistsError(output)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(
        table["day_offset"],
        table["with_online"],
        color="#2986ad",
        label="With online learning",
        linewidth=1.8,
    )
    ax.plot(
        table["day_offset"],
        table["without_online"],
        color="#d8a24a",
        label="Without online learning",
        linewidth=1.8,
    )
    ax.axhline(0, color="#b7bbc1", linewidth=0.7)
    if scored_start is not None:
        ax.axvline(
            scored_start - start,
            color="#9198a1",
            linestyle="--",
            linewidth=0.8,
            label=f"Scored period begins: {scored_start}",
        )
    ax.set(
        title=title,
        xlabel=f"Days since date_id {start}",
        ylabel=f"Rolling {window}-day weighted zero-mean R²",
    )
    ax.set_xlim(0, int(offline["date_id"][-1]) - start)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(paths["png"], dpi=180)
    fig.savefig(paths["svg"])
    plt.close(fig)
    table.write_csv(paths["csv"])
    return paths
