import numpy as np
import polars as pl
import torch
from test_patrick_pipeline import panel, tiny_config

from src.data.loader import FrameSource
from src.data.schema import FEATURES, RESPONDERS
from src.training.patrick import prepare_training, train_model


def test_future_partition_cannot_change_preprocessing_targets_or_trained_model(tmp_path):
    original = panel()
    changed = original.with_columns(
        [
            pl.when(pl.col("date_id") >= 3).then(pl.col(c) * -10000).otherwise(pl.col(c)).alias(c)
            for c in FEATURES + RESPONDERS
        ]
    )
    config = tiny_config()
    prepared = [
        prepare_training(FrameSource(frame), (0, 1, 2), config.features, tmp_path / str(i))
        for i, frame in enumerate([original, changed])
    ]
    assert prepared[0].features.state_dict() == prepared[1].features.state_dict()
    for date in (0, 1, 2):
        for a, b in zip(prepared[0].read(date), prepared[1].read(date), strict=True):
            np.testing.assert_array_equal(a, b)
    models = [train_model(p, config.model, training=config.training, seed=2)[0] for p in prepared]
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)


def test_source_is_never_read_outside_training_partition(tmp_path):
    class GuardedSource(FrameSource):
        def day(self, date):
            assert date in (0, 1, 2), "future training read"
            return super().day(date)

    prepare_training(GuardedSource(panel()), (0, 1, 2), tiny_config().features, tmp_path / "cache")
