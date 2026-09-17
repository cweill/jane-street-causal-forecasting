import numpy as np
import polars as pl
import pytest


@pytest.fixture
def panel():
    rows = []
    rng = np.random.default_rng(7)
    for date in range(7):
        for time in range(5):
            for symbol in (2, 9):
                row = {
                    "date_id": date,
                    "time_id": time,
                    "symbol_id": symbol,
                    "weight": 1.0 + symbol,
                    "is_scored": date >= 3,
                }
                row.update({f"feature_{i:02d}": float(rng.normal()) for i in range(79)})
                row.update(
                    {f"responder_{i}": float(date * 100 + time * 10 + symbol + i) for i in range(9)}
                )
                rows.append(row)
    return pl.DataFrame(rows)
