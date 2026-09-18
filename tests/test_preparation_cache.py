import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import pytest
from test_patrick_pipeline import panel, tiny_config

from src.data.loader import FrameSource


def api():
    assert importlib.util.find_spec("src.training.cache"), "persistent preparation cache missing"
    return importlib.import_module("src.training.cache")


class TrainingSource:
    def __init__(self):
        self.source = FrameSource(panel())
        self.allow_reads = True

    def dates(self):
        return self.source.dates()

    def day(self, date):
        assert self.allow_reads, "cache hit reread raw training data"
        assert date in (0, 1, 2), "preparation read future data"
        return self.source.day(date)


def test_cache_reuse_loads_identical_arrays_and_frozen_preprocessing_without_raw_reads(tmp_path):
    module, source, config = api(), TrainingSource(), tiny_config()
    first, a = module.prepare_cached(source, (0, 1, 2), config.features, tmp_path, "a" * 64)
    source.allow_reads = False
    second, b = module.prepare_cached(source, (0, 1, 2), config.features, tmp_path, "a" * 64)
    assert not a["hit"] and b["hit"] and a["key"] == b["key"]
    assert first.features.state_dict() == second.features.state_dict()
    for x, y in zip(first.read(1), second.read(1), strict=True):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="outside"):
        second.read(3)


def test_cache_invalidates_for_dataset_dates_features_and_code(tmp_path, monkeypatch):
    module, source, config = api(), TrainingSource(), tiny_config()
    _, baseline = module.prepare_cached(source, (0, 1, 2), config.features, tmp_path, "a" * 64)
    variants = [
        ((0, 1), config.features, "a" * 64),
        ((0, 1, 2), replace(config.features, time_steps=1000), "a" * 64),
        ((0, 1, 2), config.features, "b" * 64),
    ]
    for dates, features, digest in variants:
        _, record = module.prepare_cached(source, dates, features, tmp_path, digest)
        assert not record["hit"] and record["key"] != baseline["key"]
    monkeypatch.setattr(module, "preprocessing_fingerprint", lambda: "changed-code")
    _, record = module.prepare_cached(source, (0, 1, 2), config.features, tmp_path, "a" * 64)
    assert not record["hit"] and record["key"] != baseline["key"]


def test_cache_rejects_corrupt_arrays_instead_of_training_on_them(tmp_path):
    module, source, config = api(), TrainingSource(), tiny_config()
    prepared, _ = module.prepare_cached(source, (0, 1, 2), config.features, tmp_path, "a" * 64)
    with (prepared.directory / "1.npz").open("ab") as handle:
        handle.write(b"corruption")
    source.allow_reads = False
    with pytest.raises(ValueError, match="checksum"):
        module.prepare_cached(source, (0, 1, 2), config.features, tmp_path, "a" * 64)
