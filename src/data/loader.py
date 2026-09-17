"""Read one chronological day at a time, including partitioned competition parquet."""

from pathlib import Path
from typing import Protocol

import polars as pl

from src.data.schema import validate_day


class DaySource(Protocol):
    def dates(self) -> tuple[int, ...]: ...
    def day(self, date: int) -> pl.DataFrame: ...


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
