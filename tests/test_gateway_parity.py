import json
from pathlib import Path

import polars as pl

from scripts.verify_gateway import fixture_panel, trace_batch
from src.data.api_simulator import APISimulator
from src.data.loader import FrameSource


def test_replay_matches_trace_from_unmodified_distributed_gateway():
    fixture = json.loads((Path(__file__).parent / "fixtures/gateway_trace.json").read_text())
    actual = []

    def predict(test, lags):
        actual.append(trace_batch(test, lags))
        return test.select("row_id").with_columns(pl.lit(0.0).alias("responder_6"))

    APISimulator(FrameSource(fixture_panel()), range(4)).run(predict)
    assert actual == fixture["batches"]
