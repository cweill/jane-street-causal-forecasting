import copy

import polars as pl
import torch
from test_patrick_pipeline import train

from src.artifacts import load_predictor
from src.data.api_simulator import APISimulator


def test_phase_timing_preserves_replay_and_online_state(tmp_path):
    from scripts.profile_patrick_scaling import timed_replay, training_ranges

    source, _, _ = train(tmp_path)
    reference = load_predictor(tmp_path / "model")
    reference.stacked_inference = True
    timed = copy.deepcopy(reference)
    expected = APISimulator(source, [3, 4, 5]).run(reference.predict).predictions
    with training_ranges():
        records, outputs = timed_replay(timed, source, [3, 4, 5])
    assert pl.concat(outputs).equals(expected)
    assert [r["optimizer_steps"] for r in records] == [0, 3, 3]
    for record in records:
        assert record["total_seconds"] >= record["update_seconds"] >= record["panel_seconds"]
        assert record["inference_seconds"] >= record["stack_seconds"] >= 0
        assert record["other_seconds"] >= 0
    assert reference.update_log == timed.update_log
    for a, b in zip(reference.models, timed.models, strict=True):
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    for a, b in zip(reference._optimizers, timed._optimizers, strict=True):
        for key, state in a.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, b.state_dict()["state"][key][name], rtol=0, atol=0
                )
