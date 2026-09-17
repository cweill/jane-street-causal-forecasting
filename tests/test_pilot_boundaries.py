import polars as pl
import pytest

from scripts.real_data_pilot import DATES, validate_panel


def test_pilot_rejects_holdout_even_when_required_dates_are_present():
    frame = pl.DataFrame({"date_id": [*DATES, 1499]})
    with pytest.raises(ValueError, match="holdout forbidden"):
        validate_panel(frame)


def test_pilot_rejects_missing_day_instead_of_shifting_split():
    frame = pl.DataFrame({"date_id": list(DATES[:-1])})
    with pytest.raises(ValueError, match="exactly dates"):
        validate_panel(frame)
