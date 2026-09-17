"""Explicit allowlists prevent responder-derived columns entering model inputs."""

import numpy as np
import polars as pl

KEYS = ["date_id", "time_id", "symbol_id"]
FEATURES = [f"feature_{i:02d}" for i in range(79)]
RESPONDERS = [f"responder_{i}" for i in range(9)]
LAG_COLUMNS = [f"{r}_lag_1" for r in RESPONDERS]
TEST_COLUMNS = ["row_id", *KEYS, "weight", "is_scored", *FEATURES]


def validate_day(frame: pl.DataFrame):
    missing = set(KEYS + ["weight"] + FEATURES + RESPONDERS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing historical columns: {sorted(missing)}")
    if frame.is_empty():
        return
    if frame["date_id"].n_unique() != 1:
        raise ValueError("expected one date")
    for key in KEYS:
        if not frame[key].dtype.is_integer() or frame[key].null_count() or frame[key].min() < 0:
            raise ValueError(f"{key} must be nonnegative integers")
    if frame.select(KEYS).is_duplicated().any():
        raise ValueError("duplicate (date_id, time_id, symbol_id)")
    for col in ["weight", *RESPONDERS]:
        if not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError(f"{col} must be finite")
    if frame["weight"].min() < 0:
        raise ValueError("weights must be nonnegative")
    if "is_scored" in frame.columns and (
        frame["is_scored"].dtype != pl.Boolean or frame["is_scored"].null_count()
    ):
        raise ValueError("is_scored must be boolean and non-null")


def validate_test(frame: pl.DataFrame):
    if set(frame.columns) != set(TEST_COLUMNS):
        raise ValueError("test schema must contain only API-visible columns; responder leak")
    if frame.is_empty() or frame.select("date_id", "time_id").n_unique() != 1:
        raise ValueError("expected one nonempty (date_id, time_id) batch")
    if frame["symbol_id"].n_unique() != frame.height:
        raise ValueError("duplicate symbol in batch")


def test_view(day: pl.DataFrame, row_offset=0):
    if "is_scored" not in day.columns:
        day = day.with_columns(pl.lit(True).alias("is_scored"))
    if "row_id" not in day.columns:
        day = day.with_columns(pl.Series("row_id", np.arange(row_offset, row_offset + day.height)))
    return day.select(TEST_COLUMNS)
