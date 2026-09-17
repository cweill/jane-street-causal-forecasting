"""Small deterministic panel with missing observations and new symbols; no score claims."""

import numpy as np
import polars as pl


def synthetic_panel(days=10, times=12, symbols=3, seed=17):
    if days < 3 or times < 2 or symbols < 1:
        raise ValueError("synthetic panel requires days>=3, times>=2, symbols>=1")
    rng = np.random.default_rng(seed)
    rows = []
    for date in range(days):
        for time in range(times):
            for symbol in range(symbols + (date >= days - 2)):
                if symbol == 1 and time == 2:
                    continue
                features = rng.normal(size=79)
                target = 0.15 * features[6] + 0.05 * np.sin(time) + rng.normal(scale=0.4)
                row = {
                    "date_id": date,
                    "time_id": time,
                    "symbol_id": symbol,
                    "weight": float(0.5 + rng.random()),
                    "is_scored": date >= 2,
                }
                row.update({f"feature_{i:02d}": float(v) for i, v in enumerate(features)})
                row.update(
                    {f"responder_{i}": float(target + rng.normal(scale=0.2)) for i in range(9)}
                )
                rows.append(row)
    return pl.DataFrame(rows)
