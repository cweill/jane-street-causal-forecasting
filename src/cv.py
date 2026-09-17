"""Deterministic expanding or rolling training windows, split only at date boundaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalFold:
    index: int
    train_dates: tuple[int, ...]
    warmup_dates: tuple[int, ...]
    validation_dates: tuple[int, ...]

    @property
    def replay_dates(self):
        return self.warmup_dates + self.validation_dates


def temporal_folds(
    dates, n_splits=2, validation_days=200, gap_days=0, min_train_days=1, max_train_days=None
):
    dates = tuple(sorted({int(d) for d in dates}))
    if n_splits < 1 or validation_days < 1 or gap_days < 0 or min_train_days < 1:
        raise ValueError("invalid temporal split sizes")
    if max_train_days is not None and max_train_days < min_train_days:
        raise ValueError("max_train_days must be at least min_train_days")
    first = len(dates) - n_splits * validation_days
    if first - gap_days < min_train_days:
        raise ValueError("insufficient dates for requested temporal folds")
    folds = []
    for index in range(n_splits):
        start = first + index * validation_days
        stop = start - gap_days
        train_start = 0 if max_train_days is None else max(0, stop - max_train_days)
        folds.append(
            TemporalFold(
                index,
                dates[train_start:stop],
                dates[stop:start],
                dates[start : start + validation_days],
            )
        )
    return folds
