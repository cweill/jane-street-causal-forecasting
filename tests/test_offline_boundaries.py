import numpy as np
import polars as pl
import torch

from src.data.features import FeatureConfig
from src.data.loader import FrameSource
from src.data.schema import FEATURES, RESPONDERS
from src.models.grigoreva_gru import ModelConfig
from src.training.offline import prepare_training, train_model


def test_future_partition_cannot_change_scaler_features_aux_labels_or_trained_model(
    panel, tmp_path
):
    changed = panel.with_columns(
        [
            pl.when(pl.col("date_id") >= 3).then(pl.col(c) * -10000).otherwise(pl.col(c)).alias(c)
            for c in FEATURES + RESPONDERS
        ]
    )
    sources = [FrameSource(panel), FrameSource(changed)]
    prepared = [
        prepare_training(s, (0, 1, 2), FeatureConfig(True, True, 3), tmp_path / str(i))
        for i, s in enumerate(sources)
    ]
    np.testing.assert_array_equal(prepared[0].scaler.mean, prepared[1].scaler.mean)
    for date in (0, 1, 2):
        a, b = [p.read(date) for p in prepared]
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    # Last 4 observations per symbol require censored r9; no validation label substitution.
    assert prepared[0].read(2)["auxiliary_mask"][:, 1].sum() == 2
    config = ModelConfig(
        auxiliary_targets=True,
        hidden_sizes=(4,),
        dropout=(0.0,),
        linear_sizes=(),
        linear_dropout=(),
    )
    models = [train_model(p, config, seed=2)[0] for p in prepared]
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)


def test_source_is_never_read_outside_training_partition(panel, tmp_path):
    class GuardedSource(FrameSource):
        def day(self, date):
            assert date in (0, 1, 2), "future training read"
            return super().day(date)

    prepare_training(GuardedSource(panel), (0, 1, 2), FeatureConfig(), tmp_path / "cache")
