"""Differential check against the distributed gateway's actual batching method.

Usage: python scripts/verify_gateway.py --gateway-source /path/jane_street_gateway.py
The method is extracted without changing its body, avoiding unrelated RPC dependencies.
Only the pinned, inspected source is executed. Transport/timeouts are outside this check.
"""

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource
from src.data.synthetic import synthetic_panel


def fixture_panel():
    return (
        synthetic_panel(days=4, times=4, symbols=2)
        .with_row_index("row_id")
        .filter(~((pl.col("date_id") == 2) & (pl.col("time_id") == 0)))
    )


def trace_batch(test, lags):
    value = {"test": test.to_dicts(), "lags": None if lags is None else lags.to_dicts()}
    return {
        "date": int(test["date_id"][0]),
        "time": int(test["time_id"][0]),
        "rows": test.height,
        "lag_rows": None if lags is None else lags.height,
        "sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
    }


def compare(gateway_path):
    content = Path(gateway_path).read_text()
    source_sha = hashlib.sha256(content.encode()).hexdigest()
    inspected_hashes = {
        # Previously inspected mirror.
        "d1aa2ab405f0d9b7e74a7c612e657ebcd632a4ab86e7fa8b80e517051e243984",
        # Official Kaggle download, 2026-09-17: only import ordering differs.
        "d7e6fe922399999df152ba8bfd09431ed95716c71a767269190eefd20adc8927",
    }
    if source_sha not in inspected_hashes:
        raise ValueError("gateway source changed; inspect it before updating the pin")
    tree = ast.parse(content)
    klass = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JSGateway"
    )
    method = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_data_batches"
    )
    namespace = {"pl": pl, "os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(gateway_path), "exec"), namespace)  # noqa: S102 -- pinned, inspected reference method
    historical = fixture_panel()
    expected = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for date in range(4):
            day = historical.filter(pl.col("date_id") == date)
            test = day.drop([f"responder_{i}" for i in range(9)])
            # Independently construct the gateway's on-disk input convention.
            lags = historical.filter(pl.col("date_id") == date - 1).select(
                "date_id", "time_id", "symbol_id", *[f"responder_{i}" for i in range(9)]
            )
            lags = lags.with_columns(pl.lit(date, dtype=pl.Int64).alias("date_id")).rename(
                {f"responder_{i}": f"responder_{i}_lag_1" for i in range(9)}
            )
            for name, frame in (("test", test), ("lags", lags)):
                directory = root / name / f"date_id={date}"
                directory.mkdir(parents=True)
                frame.write_parquet(directory / "part.parquet")
        gateway = SimpleNamespace(test_path=str(root / "test"), lags_path=str(root / "lags"))
        for (test, lags), validation in namespace["generate_data_batches"](gateway):
            assert validation.equals(test.select("row_id"))
            expected.append(trace_batch(test, lags))
    actual = []

    def predict(test, lags):
        actual.append(trace_batch(test, lags))
        return test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))

    APISimulator(FrameSource(historical), range(4)).run(predict)
    if actual != expected:
        raise AssertionError(f"gateway mismatch\nexpected={expected}\nactual={actual}")
    return {"gateway_sha256": source_sha, "batches": expected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-source", required=True)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    result = compare(args.gateway_source)
    if args.write_fixture:
        output = Path("tests/fixtures/gateway_trace.json")
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Matched all {len(result['batches'])} batches against gateway {result['gateway_sha256']}"
    )
