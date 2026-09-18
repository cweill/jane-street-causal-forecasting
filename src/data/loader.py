"""Read one chronological day at a time, including partitioned competition parquet."""

from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import polars as pl

from src.data.schema import validate_day


class DaySource(Protocol):
    def dates(self) -> tuple[int, ...]: ...
    def day(self, date: int) -> pl.DataFrame: ...


class CachedDaySource:
    """Evaluator-owned cache of immutable days; never handed to a predictor."""

    def __init__(self, source, capacity=2):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("positive day cache capacity required")
        self._source, self._capacity = source, capacity
        self._dates = source.dates()
        self._cache = OrderedDict()

    def dates(self):
        return self._dates

    def day(self, date):
        if date not in self._cache:
            self._cache[date] = self._source.day(date).clone()
        self._cache.move_to_end(date)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return self._cache[date].clone()


class RestrictedDateSource:
    """Reject out-of-partition reads before reaching the underlying source."""

    def __init__(self, source, dates):
        self._source = source
        self._dates = tuple(sorted(set(dates)))
        self._allowed = set(self._dates)
        if not self._dates or not self._allowed.issubset(source.dates()):
            raise ValueError("restricted source dates missing")

    def dates(self):
        return self._dates

    def day(self, date):
        if date not in self._allowed:
            raise ValueError(f"date {date} outside declared partition")
        return self._source.day(date)


class FrameSource:
    def __init__(self, frame: pl.DataFrame):
        self._frame = frame.clone()

    def dates(self):
        return tuple(sorted(self._frame["date_id"].unique().to_list()))

    def day(self, date):
        frame = self._frame.filter(pl.col("date_id") == date).sort("time_id", "symbol_id")
        validate_day(frame)
        return frame


class ParquetSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        paths = sorted(self.path.rglob("*.parquet")) if self.path.is_dir() else [self.path]
        if not paths or any(not p.is_file() for p in paths):
            raise FileNotFoundError(f"no parquet files at {path}")
        self._scan = pl.scan_parquet(paths, hive_partitioning=True)

    def dates(self):
        return tuple(self._scan.select(pl.col("date_id").unique().sort()).collect()["date_id"])

    def day(self, date):
        frame = self._scan.filter(pl.col("date_id") == date).collect().sort("time_id", "symbol_id")
        validate_day(frame)
        return frame
