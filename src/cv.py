"""Deterministic expanding or rolling training windows, split only at date boundaries."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TemporalFold:
    index: int
    train_dates: tuple[int, ...]
    warmup_dates: tuple[int, ...]
    validation_dates: tuple[int, ...]

    @property
    def replay_dates(self):
        return self.warmup_dates + self.validation_dates


def configured_folds(dates, config):
    """Resolve fixed calendar boundaries, or retain the legacy tail-count splitter.

    A fixed protocol requires every declared date. Extra source dates are ignored;
    missing dates never shorten or shift its training, warmup or scored intervals.
    """
    options = asdict(config)
    start = options.pop("min_date")
    boundaries = [options.pop(k) for k in ("train_end", "warmup_end", "validation_end")]
    if all(value is None for value in boundaries):
        return temporal_folds([d for d in dates if d >= start], **options)
    if any(type(value) is not int for value in [start, *boundaries]):
        raise ValueError("fixed protocols require all three integer boundaries")
    train_end, warmup_end, validation_end = boundaries
    if not 0 <= start <= train_end <= warmup_end < validation_end:
        raise ValueError("invalid fixed protocol ordering")
    if (
        config.n_splits != 1
        or config.max_train_days is not None
        or config.gap_days != warmup_end - train_end
        or config.validation_days != validation_end - warmup_end
        or train_end - start + 1 < config.min_train_days
    ):
        raise ValueError("split counts conflict with fixed boundaries")
    required = set(range(start, validation_end + 1))
    missing = sorted(required - set(dates))
    if missing:
        raise ValueError(f"fixed protocol dates missing: {missing[:10]}")
    return [
        TemporalFold(
            0,
            tuple(range(start, train_end + 1)),
            tuple(range(train_end + 1, warmup_end + 1)),
            tuple(range(warmup_end + 1, validation_end + 1)),
        )
    ]


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
