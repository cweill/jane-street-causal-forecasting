"""Scalar-only observations of committed artifacts; never called by the predictor."""

import json
import math
from itertools import pairwise
from pathlib import Path


def training_events(saved, *, days_per_epoch, epochs):
    completed, epoch, position = saved["completed"], saved["epoch"], saved["position"]
    if (
        days_per_epoch < 1
        or epochs < 1
        or completed != epoch * days_per_epoch + position
        or not 0 <= position < days_per_epoch
        or not 0 <= completed <= days_per_epoch * epochs
        or len(saved["losses"]) != position
    ):
        raise ValueError("inconsistent training checkpoint counters")
    events = []
    for record in saved["history"]:
        loss, number = float(record["mean_optimization_loss"]), int(record["epoch"])
        if not math.isfinite(loss) or not 1 <= number <= epoch:
            raise ValueError("invalid completed epoch metrics")
        events.append(
            {
                "step": 2 * number * days_per_epoch + 1,
                "metrics": {"train/epoch": number, "train/epoch_mean_optimization_loss": loss},
            }
        )
    start = completed - len(saved["losses"]) + 1
    for batch, loss in enumerate(saved["losses"], start=start):
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError("non-finite recorded optimization loss")
        events.append(
            {
                "step": 2 * batch,
                "metrics": {
                    "train/batch": batch,
                    "train/optimization_loss": loss,
                    "train/epoch_fraction": batch / days_per_epoch,
                    "train/progress_fraction": batch / (days_per_epoch * epochs),
                },
            }
        )
    return sorted(events, key=lambda event: event["step"])


def evaluation_events(root, mode, *, replay_dates, total_batches, window=20):
    if mode not in ("offline", "online") or window < 1:
        raise ValueError("invalid evaluation mode/window")
    dates = tuple(replay_dates)
    if not dates or any(b != a + 1 for a, b in pairwise(dates)):
        raise ValueError("consecutive replay dates required")
    base = 2 * total_batches + 2 + (len(dates) if mode == "online" else 0)
    directory = Path(root) / mode
    events, recent = [], []
    scored_sse, scored_energy = 0.0, 0.0
    for offset, date in enumerate(dates):
        marker = directory / "checkpoints" / f"date_{date}" / "replay.json"
        path = directory / f"date_{date}" / "result.json"
        # A result written before its checkpoint is not yet a committed replay day.
        if not marker.exists() or not path.exists():
            break
        if json.loads(marker.read_text())["completed_date"] != date:
            raise ValueError("replay checkpoint date mismatch")
        record = json.loads(path.read_text())
        if record["date_id"] != date:
            raise ValueError("daily metric date mismatch")
        diagnostic, primary = record["diagnostic"], record["primary"]
        values = [float(s[k]) for s in (diagnostic, primary) for k in ("sse", "denominator")]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("invalid daily sufficient statistics")
        sse, energy, primary_sse, primary_energy = values
        recent.append((sse, energy))
        scored_sse += primary_sse
        scored_energy += primary_energy
        metrics = {
            "eval/date_id": date,
            f"{mode}/days_completed": offset + 1,
            f"{mode}/is_scored": bool(primary["rows"]),
            f"{mode}/day_seconds": float(record["seconds"]),
        }
        if energy > 0:
            metrics[f"{mode}/diagnostic_daily_r2"] = 1 - sse / energy
        if scored_energy > 0:
            metrics[f"{mode}/scored_cumulative_r2"] = 1 - scored_sse / scored_energy
        if len(recent) >= window:
            numerator = sum(p[0] for p in recent[-window:])
            denominator = sum(p[1] for p in recent[-window:])
            if denominator > 0:
                metrics[f"{mode}/rolling_r2"] = 1 - numerator / denominator
        events.append({"step": base + offset, "metrics": metrics})
    return events
