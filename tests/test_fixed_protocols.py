from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_patrick_pipeline import train

from src import cv
from src.artifacts import load_predictor
from src.config import load_config
from src.data import loader
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource


def fixed_config(name):
    path = Path(f"configs/patrick_{name}.yaml")
    assert path.exists(), "fixed Patrick protocol has not been added"
    return load_config(path)


def folds(dates, config):
    assert hasattr(cv, "configured_folds"), "explicit date boundary resolver not implemented"
    return cv.configured_folds(dates, config.cv)


@pytest.mark.parametrize(
    "name,train_end,warm_start,warm_end,score_start,score_end",
    [("development", 1059, 1060, 1179, 1180, 1379), ("final", 1379, 1380, 1499, 1500, 1698)],
)
def test_protocol_boundaries_do_not_move_when_later_dates_are_added(
    name, train_end, warm_start, warm_end, score_start, score_end
):
    config = fixed_config(name)
    short = folds(range(score_end + 1), config)
    full = folds(list(range(1800))[::-1] + [1000], config)
    assert short == full
    (fold,) = full
    assert fold.train_dates == tuple(range(train_end + 1))
    assert fold.warmup_dates == tuple(range(warm_start, warm_end + 1))
    assert fold.validation_dates == tuple(range(score_start, score_end + 1))
    assert set(fold.train_dates).isdisjoint(fold.replay_dates)


@pytest.mark.parametrize("missing", [0, 1059, 1060, 1179, 1180, 1379])
def test_missing_protocol_day_fails_instead_of_shortening_or_shifting(missing):
    config = fixed_config("development")
    with pytest.raises(ValueError, match="missing"):
        folds([d for d in range(1699) if d != missing], config)


def test_conflicting_counts_or_partial_boundaries_are_rejected():
    config = fixed_config("development")
    for changes in (
        {"validation_days": 199},
        {"n_splits": 2},
        {"train_end": None},
        {"max_train_days": 10},
        {"warmup_end": 1058},
    ):
        with pytest.raises(ValueError):
            folds(range(1699), replace(config, cv=replace(config.cv, **changes)))


def test_restricted_source_rejects_future_reads_before_accessing_underlying_data(panel):
    assert hasattr(loader, "RestrictedDateSource"), "date read boundary not implemented"
    source = FrameSource(panel)
    restricted = loader.RestrictedDateSource(source, (0, 1, 2))
    assert restricted.dates() == (0, 1, 2)
    assert restricted.day(2).equals(source.day(2))
    with pytest.raises(ValueError, match="outside"):
        restricted.day(3)
    with pytest.raises(ValueError, match="missing"):
        loader.RestrictedDateSource(source, (0, 99))


@pytest.mark.parametrize("start", [1060, 1380])
def test_first_replay_lag_is_visible_but_first_update_needs_cached_inputs(tmp_path, start):
    source, _, _ = train(tmp_path)
    frame = pl.concat([source.day(d) for d in source.dates()]).with_columns(
        (pl.col("date_id") + start - 3).alias("date_id")
    )
    adapted, frozen = load_predictor(tmp_path / "model"), load_predictor(tmp_path / "model")
    frozen.config = replace(frozen.config, enabled=False)
    releases = []

    def observe(test, lags):
        if lags is not None:
            releases.append((int(test["date_id"][0]), lags.height))
        return adapted.predict(test, lags)

    online = APISimulator(FrameSource(frame), [start, start + 1]).run(observe).predictions
    offline = APISimulator(FrameSource(frame), [start, start + 1]).run(frozen.predict).predictions
    np.testing.assert_array_equal(
        online.filter(pl.col("date_id") == start)["responder_6"],
        offline.filter(pl.col("date_id") == start)["responder_6"],
    )
    assert releases[0] == (start, source.day(2).height)
    assert [(u["released_at"], u["source_date"]) for u in adapted.update_log] == [
        ([start + 1, 0], start)
    ]
