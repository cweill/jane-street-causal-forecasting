import numpy as np
import pytest

from src.data.normalization import TrainOnlyStandardizer


def test_scaler_rejects_validation_fit_and_freezes():
    scaler = TrainOnlyStandardizer((0, 1), 2)
    scaler.update(np.array([[1.0, np.nan], [3.0, np.nan]]), date=0)
    with pytest.raises(ValueError, match="training"):
        scaler.update(np.ones((2, 2)), date=2)
    scaler.freeze()
    np.testing.assert_allclose(scaler.transform(np.array([[2.0, np.nan]])), [[0.0, 0.0]])
    with pytest.raises(ValueError, match="frozen"):
        scaler.update(np.ones((2, 2)), date=1)
