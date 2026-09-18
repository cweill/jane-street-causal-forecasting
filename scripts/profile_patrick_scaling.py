"""Read-only timing instrumentation for the existing causal Patrick replay path."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from time import perf_counter
from unittest.mock import patch

import torch

from src.data.api_simulator import APISimulator
from src.models.patrick_ensemble import StackedPatrick
from src.models.patrick_yam import PatrickYam
from src.training import patrick


def synchronize(predictor):
    device = next(predictor.models[0].parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_replay(predictor, source, dates):
    """Nested phase wall times; inference ends with the existing CPU output copy.

    GPU synchronization occurs only at day/update boundaries, never per kernel.
    Panel preparation is INCLUDED in updates and stacking is INCLUDED in inference.
    """
    seconds = defaultdict(float)
    inside_update = False

    def timer(name, function):
        @wraps(function)
        def run(*args, **kwargs):
            start = perf_counter()
            result = function(*args, **kwargs)
            seconds[name] += perf_counter() - start
            return result

        return run

    original_update = predictor._update

    def update(*args, **kwargs):
        nonlocal inside_update
        synchronize(predictor)
        start = perf_counter()
        inside_update = True
        try:
            result = original_update(*args, **kwargs)
            synchronize(predictor)
            return result
        finally:
            inside_update = False
            seconds["update_seconds"] += perf_counter() - start

    original_features = predictor.features._transform_rows

    def features(*args, **kwargs):
        start = perf_counter()
        result = original_features(*args, **kwargs)
        if not inside_update:
            seconds["feature_seconds"] += perf_counter() - start
        return result

    records, outputs, offset = [], [], 0
    with (
        patch.object(predictor, "_update", update),
        patch.object(predictor.features, "_transform_rows", features),
        patch.object(patrick, "pack_panel", timer("panel_seconds", patrick.pack_panel)),
        patch.object(StackedPatrick, "refresh", timer("stack_seconds", StackedPatrick.refresh)),
        patch.object(
            predictor, "_predict_stacked", timer("inference_seconds", predictor._predict_stacked)
        ),
    ):
        for date in dates:
            seconds.clear()
            before = len(predictor.update_log)
            synchronize(predictor)
            start = perf_counter()
            result = APISimulator(source, [date], row_offset=offset, prepartition_truth=True).run(
                predictor.predict
            )
            synchronize(predictor)
            elapsed = perf_counter() - start
            record = {
                key: seconds[key]
                for key in (
                    "update_seconds",
                    "panel_seconds",
                    "stack_seconds",
                    "inference_seconds",
                    "feature_seconds",
                )
            }
            record.update(
                date_id=date,
                total_seconds=elapsed,
                timestamps=result.calls,
                models=len(predictor.models),
                optimizer_steps=(len(predictor.update_log) - before)
                * len(predictor.models)
                * predictor.config.steps,
                other_seconds=elapsed
                - record["update_seconds"]
                - record["inference_seconds"]
                - record["feature_seconds"],
            )
            outputs.append(result.predictions)
            offset += result.predictions.height
            records.append(record)
    return records, outputs


@contextmanager
def training_ranges():
    """Annotate actual training calls for torch.profiler; never replace calculations."""

    def marked(name, function):
        @wraps(function)
        def run(*args, **kwargs):
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)

        return run

    targets = [
        (patrick, "pack_panel", "phase/panel_preparation"),
        (PatrickYam, "forward", "phase/forward"),
        (patrick, "loss_for_model", "phase/loss"),
        (torch.Tensor, "backward", "phase/backward"),
        (torch.nn.utils, "clip_grad_norm_", "phase/gradient_clip"),
        (torch.optim.Adam, "step", "phase/adam_step"),
    ]
    with ExitStack() as stack:
        for owner, attribute, label in targets:
            stack.enter_context(
                patch.object(owner, attribute, marked(label, getattr(owner, attribute)))
            )
        yield
