import numpy as np
import polars as pl
import pytest

from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.data.schema import FEATURES, RESPONDERS


def test_api_lags_are_entire_previous_day_stamped_current_day(panel):
    seen = []

    def predict(test, lags):
        d, t = test["date_id"][0], test["time_id"][0]
        assert set(test.columns) == set(
            FEATURES + ["row_id", "date_id", "time_id", "symbol_id", "weight", "is_scored"]
        )
        if t == 0:
            assert lags is not None
            assert lags["date_id"].unique().to_list() == [d]
            expected = panel.filter(pl.col("date_id") == d - 1)
            assert lags.height == expected.height
            assert lags["responder_6_lag_1"].to_list() == expected["responder_6"].to_list()
            assert not any(c in lags.columns for c in RESPONDERS)
        else:
            assert lags is None
        seen.append((d, t))
        return test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))

    result = APISimulator(FrameSource(panel), [2, 3]).run(predict)
    assert len(seen) == 10
    assert result.metric.rows == 10  # unscored day was still fully served
    assert result.metric.score == 0


def test_missing_previous_date_is_empty_lags_not_previous_observed_day(panel):
    source = FrameSource(panel.filter(pl.col("date_id") != 2))

    def predict(test, lags):
        if test["time_id"][0] == 0:
            assert lags is not None and lags.is_empty()
        return test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))

    APISimulator(source, [3]).run(predict)


def test_no_release_when_day_does_not_contain_time_zero(panel):
    source = FrameSource(panel.filter(~((pl.col("date_id") == 3) & (pl.col("time_id") == 0))))

    def predict(test, lags):
        assert lags is None
        return test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))

    APISimulator(source, [3]).run(predict)


@pytest.mark.parametrize("kind", ["reordered", "nan", "missing", "extra"])
def test_prediction_contract_is_strict(panel, kind):
    def bad(test, lags):
        out = test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))
        if kind == "reordered":
            return out.reverse()
        if kind == "nan":
            return out.with_columns(pl.lit(np.nan).alias("responder_6"))
        if kind == "extra":
            return out.with_columns(pl.lit(0).alias("secret"))
        return out.head(1)

    with pytest.raises(ValueError):
        APISimulator(FrameSource(panel), [3]).run(bad)
