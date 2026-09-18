"""Scalar-only observations of committed artifacts; never called by the predictor."""

import json
import math
from itertools import pairwise
from pathlib import Path


def _optional_diagnostic(metrics, name, value):
    if value is not None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite training diagnostic")
        metrics[name] = value


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
    diagnostics = saved.get("diagnostics")
    if diagnostics is not None and len(diagnostics) != position:
        raise ValueError("inconsistent training diagnostic counters")
    events = []
    for record in saved["history"]:
        loss, number = float(record["mean_optimization_loss"]), int(record["epoch"])
        if not math.isfinite(loss) or not 1 <= number <= epoch:
            raise ValueError("invalid completed epoch metrics")
        metrics = {"train/epoch": number, "train/epoch_mean_optimization_loss": loss}
        _optional_diagnostic(
            metrics, "train/epoch_mean_unbalanced_loss", record.get("mean_unbalanced_loss")
        )
        _optional_diagnostic(metrics, "train/epoch_responder_6_r2", record.get("responder_6_r2"))
        events.append(
            {
                "step": 2 * number * days_per_epoch + 1,
                "metrics": metrics,
            }
        )
    start = completed - len(saved["losses"]) + 1
    for batch, loss in enumerate(saved["losses"], start=start):
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError("non-finite recorded optimization loss")
        metrics = {
            "train/batch": batch,
            "train/optimization_loss": loss,
            "train/epoch_fraction": batch / days_per_epoch,
            "train/progress_fraction": batch / (days_per_epoch * epochs),
        }
        if diagnostics is not None:
            record = diagnostics[batch - start]
            sse, energy = float(record["responder_6_sse"]), float(record["responder_6_energy"])
            if not all(math.isfinite(v) and v >= 0 for v in (sse, energy)):
                raise ValueError("invalid training diagnostic statistics")
            _optional_diagnostic(metrics, "train/unbalanced_loss", record["unbalanced_loss"])
            _optional_diagnostic(
                metrics, "train/responder_6_r2", 1 - sse / energy if energy > 0 else None
            )
        events.append(
            {
                "step": 2 * batch,
                "metrics": metrics,
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
