import importlib
import importlib.util

import pytest
import torch

from src.models.patrick_yam import multitask_loss


def diagnostics():
    assert importlib.util.find_spec("src.training.diagnostics"), "training diagnostics missing"
    return importlib.import_module("src.training.diagnostics")


def test_unbalanced_error_tracks_predictions_without_changing_gradients_or_rng():
    module = diagnostics()
    target = torch.ones(1, 1, 3, 9)
    target[..., 1, :] = 3
    target[..., 2, :] = float("nan")  # Absent/zero-weight rows are excluded.
    weight = torch.tensor([[[1.0, 2.0, 0.0]]])
    prediction = torch.zeros_like(target, requires_grad=True)
    prediction.data[..., 2, :] = float("nan")
    loss = multitask_loss(prediction, target, weight, (1,) * 9)
    expected_gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    rng = torch.get_rng_state().clone()
    before = module.batch_diagnostics(prediction, target, weight, (1,) * 9)
    after = module.batch_diagnostics(target * 0.5, target, weight, (1,) * 9)
    assert before["unbalanced_loss"] == 1.0
    assert after["unbalanced_loss"] == 0.25
    assert before["responder_6_sse"] == before["responder_6_energy"] == 19.0
    assert after["responder_6_sse"] == 4.75
    assert all(type(v) is float for v in before.values())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    loss.backward()
    torch.testing.assert_close(prediction.grad, expected_gradient, rtol=0, atol=0)


def test_diagnostics_respect_auxiliary_target_switch_and_zero_energy():
    module = diagnostics()
    target = torch.ones(1, 1, 1, 9)
    prediction = torch.zeros_like(target)
    prediction[..., 6] = 0.5
    weight = torch.ones(1, 1, 1)
    main_only = (0, 0, 0, 0, 0, 0, 1, 0, 0)
    assert (
        module.batch_diagnostics(prediction, target, weight, main_only)["unbalanced_loss"] == 0.25
    )
    assert module.batch_diagnostics(prediction, target, weight, (1,) * 9)[
        "unbalanced_loss"
    ] == pytest.approx(8.25 / 9)
    empty = module.batch_diagnostics(prediction, target * 0, weight, main_only)
    assert empty["unbalanced_loss"] is None
    summary = module.epoch_diagnostics([empty])
    assert summary["mean_unbalanced_loss"] is None
    assert summary["responder_6_r2"] is None


def test_epoch_r2_pools_sse_and_energy_instead_of_averaging_daily_scores():
    module = diagnostics()
    records = [
        {"unbalanced_loss": 1.0, "responder_6_sse": 1.0, "responder_6_energy": 1.0},
        {"unbalanced_loss": 1 / 9, "responder_6_sse": 1.0, "responder_6_energy": 9.0},
    ]
    summary = module.epoch_diagnostics(records)
    assert summary["mean_unbalanced_loss"] == pytest.approx(5 / 9)
    assert summary["responder_6_r2"] == pytest.approx(0.8)
