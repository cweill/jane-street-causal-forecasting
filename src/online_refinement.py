"""Predeclared lower-rate refinement and contiguous scored-block diagnostics."""

import json
from pathlib import Path

import polars as pl

from src.online_sweep import Trial, validate_fold


def new_trials():
    return [
        Trial(f"lr_{name}_persistent", lr, False)
        for name, lr in [("1e-5", 1e-5), ("3e-5", 3e-5), ("5e-5", 5e-5), ("2e-4", 2e-4)]
    ]


def reused_trials():
    return [Trial("frozen", None, False), Trial("lr_1e-4_persistent", 1e-4, False)]


def block_comparison(roots, trials, fold, block_days=20):
    validate_fold(fold)
    if type(block_days) is not int or block_days < 1:
        raise ValueError("positive block length required")
    baseline = next(t for t in trials if t.learning_rate is None)
    rows = []
    for start in range(0, len(fold.validation_dates), block_days):
        dates = fold.validation_dates[start : start + block_days]
        totals = {}
        coverage = {}
        for trial in trials:
            stats = []
            for date in dates:
                record = json.loads(
                    (Path(roots[trial.name]) / trial.mode / f"date_{date}/result.json").read_text()
                )
                if record["date_id"] != date:
                    raise ValueError("daily coverage mismatch")
                stats.append(record["primary"])
            coverage[trial.name] = [(r["rows"], r["denominator"]) for r in stats]
            totals[trial.name] = {
                k: sum(r[k] for r in stats) for k in ("sse", "denominator", "rows")
            }
        for trial in trials:
            if coverage[trial.name] != coverage[baseline.name]:
                raise ValueError("paired block coverage mismatch")
        base = totals[baseline.name]
        if base["denominator"] <= 0:
            raise ValueError("block score undefined for zero target energy")
        frozen = 1 - base["sse"] / base["denominator"]
        for trial in trials:
            t = totals[trial.name]
            score = 1 - t["sse"] / t["denominator"]
            rows.append(
                {
                    "trial": trial.name,
                    "start_date": dates[0],
                    "end_date": dates[-1],
                    "days": len(dates),
                    "r2": score,
                    "delta_r2": score - frozen,
                    **t,
                }
            )
    return pl.DataFrame(rows)
