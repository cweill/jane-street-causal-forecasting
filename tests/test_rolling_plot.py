import importlib
import importlib.util

import numpy as np
import polars as pl
import pytest


def api():
    assert importlib.util.find_spec("src.plotting"), "rolling plot pipeline not implemented"
    return importlib.import_module("src.plotting")


def test_rolling_r2_pools_sufficient_statistics_instead_of_averaging_scores():
    daily = pl.DataFrame(
        {"date_id": [1380, 1381, 1382], "sse": [0.0, 9.0, 4.0], "denominator": [1.0, 9.0, 1.0]}
    )
    result = api().rolling_r2(daily, window=2)
    assert result["date_id"].to_list() == [1381, 1382]
    np.testing.assert_allclose(result["r2"], [0.1, -0.3], atol=1e-15)
    assert result["window_start"].to_list() == [1380, 1381]
    appended = pl.concat(
        [daily, pl.DataFrame({"date_id": [1383], "sse": [999.0], "denominator": [3.0]})]
    )
    assert api().rolling_r2(appended, window=2).head(2).equals(result)


def test_rolling_requires_complete_windows_and_rejects_missing_days():
    daily = pl.DataFrame(
        {"date_id": list(range(1380, 1400)), "sse": [1.0] * 20, "denominator": [2.0] * 20}
    )
    assert api().rolling_r2(daily.head(19)).height == 0
    result = api().rolling_r2(daily)
    assert result["date_id"].to_list() == [1399]
    assert result["r2"].to_list() == [0.5]
    with pytest.raises(ValueError, match="consecutive"):
        api().rolling_r2(daily.filter(pl.col("date_id") != 1385))
    with pytest.raises(ValueError):
        api().rolling_r2(daily.with_columns(pl.lit(0.0).alias("denominator")))


def test_diagnostic_day_score_includes_unscored_warmup_and_joins_by_key():
    keys = {"date_id": [1380, 1380], "time_id": [0, 0], "symbol_id": [0, 1]}
    truth = pl.DataFrame({**keys, "weight": [1.0, 3.0], "responder_6": [1.0, 2.0]})
    predictions = pl.DataFrame(
        {**keys, "responder_6": [1.0, 1.0], "is_scored": [False, False]}
    ).reverse()
    score = api().score_day(predictions, truth)
    assert score == {"date_id": 1380, "rows": 2, "sse": 3.0, "denominator": 13.0}
    with pytest.raises(ValueError, match="coverage"):
        api().score_day(predictions.head(1), truth)


def test_plot_exports_real_curves_with_date_alignment(tmp_path):
    daily = pl.DataFrame(
        {"date_id": list(range(1380, 1401)), "sse": [1.0] * 21, "denominator": [2.0] * 21}
    )
    online = daily.with_columns(pl.lit(0.5).alias("sse"))
    result = api().plot_online_comparison(daily, online, tmp_path / "plot", title="Test run")
    table = pl.read_csv(result["csv"])
    assert table["day_offset"].to_list() == [19, 20]
    assert table["with_online"].to_list() == [0.75, 0.75]
    assert table["without_online"].to_list() == [0.5, 0.5]
    from pathlib import Path

    assert Path(result["png"]).read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="same dates"):
        api().plot_online_comparison(daily, online.slice(1), tmp_path / "bad", title="Bad")
