"""Kaggle-style synchronous predict(test, lags) replay.

Only this evaluator sees unreleased truth. It calls the predictor once per timestamp,
validates its answer, and only then advances. This is an information-flow boundary,
not a security sandbox for hostile Python code.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import polars as pl

from src.data.loader import DaySource
from src.data.schema import KEYS, RESPONDERS, TEST_COLUMNS, test_view
from src.metric import WeightedZeroMeanR2


def previous_day_lags(previous: pl.DataFrame, current_date: int):
    if previous.height and previous["date_id"].unique().to_list() != [current_date - 1]:
        raise ValueError("lags must originate from current_date - 1")
    return (
        previous.select(KEYS + RESPONDERS)
        .with_columns(pl.lit(current_date).cast(previous["date_id"].dtype).alias("date_id"))
        .rename({c: f"{c}_lag_1" for c in RESPONDERS})
    )


@dataclass
class ReplayResult:
    metric: WeightedZeroMeanR2
    predictions: pl.DataFrame
    calls: int
    max_call_seconds: float


class APISimulator:
    def __init__(
        self, source: DaySource, dates, scored_dates=None, timeout_seconds=None, row_offset=0
    ):
        if type(row_offset) is not int or row_offset < 0:
            raise ValueError("row offset must be a nonnegative integer")
        self._row_offset = row_offset
        self._source = source
        self._dates = tuple(int(d) for d in dates)
        if not self._dates or tuple(sorted(set(self._dates))) != self._dates:
            raise ValueError("replay dates must be unique, ascending and nonempty")
        if not set(self._dates).issubset(source.dates()):
            raise ValueError("requested replay date is absent")
        self._scored_dates = None if scored_dates is None else set(scored_dates)
        self.timeout_seconds = timeout_seconds

    def run(self, predict, *, collect_predictions=True, prediction_sink=None):
        metric, outputs = WeightedZeroMeanR2(), []
        offset, calls, max_seconds = self._row_offset, 0, 0.0
        for date in self._dates:
            day = self._source.day(date)
            public = test_view(day, offset)
            if self._scored_dates is not None and date not in self._scored_dates:
                public = public.with_columns(pl.lit(False).alias("is_scored"))
            lags = previous_day_lags(self._source.day(date - 1), date)
            for test in public.partition_by("time_id", maintain_order=True):
                time = int(test["time_id"][0])
                truth = day.filter(pl.col("time_id") == time)["responder_6"].to_numpy().copy()
                # Capture evaluator-owned metadata before calling arbitrary predictor code.
                row_ids = test["row_id"].to_numpy().copy()
                weights = test["weight"].to_numpy().copy()
                scored = test["is_scored"].to_numpy().copy()
                start = perf_counter()
                output = predict(
                    test.select(TEST_COLUMNS).clone(), lags.clone() if time == 0 else None
                )
                elapsed = perf_counter() - start
                max_seconds = max(max_seconds, elapsed)
                if self.timeout_seconds is not None and elapsed > self.timeout_seconds:
                    raise TimeoutError(f"predict call exceeded {self.timeout_seconds} seconds")
                if not isinstance(output, pl.DataFrame) or set(output.columns) != {
                    "row_id",
                    "responder_6",
                }:
                    raise ValueError(
                        "prediction must be a polars frame with row_id and responder_6"
                    )
                if not np.array_equal(output["row_id"].to_numpy(), row_ids):
                    raise ValueError("prediction row_id order/coverage mismatch")
                values = output["responder_6"].to_numpy()
                if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                    raise ValueError("predictions must be finite numeric values")
                metric.update(truth, values, weights, scored)
                annotated = test.select("row_id", *KEYS, "is_scored").with_columns(
                    output["responder_6"]
                )
                if collect_predictions:
                    outputs.append(annotated)
                if prediction_sink is not None:
                    prediction_sink(annotated)
                calls += 1
            offset += day.height
        predictions = pl.concat(outputs) if outputs else pl.DataFrame()
        return ReplayResult(metric, predictions, calls, max_seconds)
