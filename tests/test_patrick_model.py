import importlib
import importlib.util

import torch


def model_api():
    assert importlib.util.find_spec("src.models.patrick_yam"), "Patrick model not implemented"
    return importlib.import_module("src.models.patrick_yam")


def setup_model():
    api = model_api()
    torch.manual_seed(18)
    config = api.PatrickModelConfig(
        d_model=8, nheads=2, d_hidden=16, layers=2, rnn_multiplier=2, head_sizes=(12, 8)
    )
    model = api.PatrickYam(config, (5, 5, 5)).eval()
    x = torch.randn(1, 5, 3, 77)
    categories = torch.randint(0, 5, (1, 5, 3, 3))
    mask = torch.ones(1, 5, 3, dtype=torch.bool)
    return model, x, categories, mask


def test_future_inputs_cannot_change_prefix_or_receive_prefix_gradient():
    model, x, categories, mask = setup_model()
    x.requires_grad_()
    prediction, _ = model(x, categories, mask)
    altered = x.detach().clone()
    altered[:, 3:] = 1234
    changed, _ = model(altered, categories, mask)
    torch.testing.assert_close(prediction[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6)
    prediction[:, :3, :, 6].sum().backward()
    assert torch.count_nonzero(x.grad[:, 3:]) == 0
    assert torch.count_nonzero(x.grad[:, :3]) > 0


def test_asset_permutation_only_permutes_predictions():
    model, x, categories, mask = setup_model()
    order = [2, 0, 1]
    expected, _ = model(x, categories, mask)
    actual, _ = model(x[:, :, order], categories[:, :, order], mask[:, :, order])
    torch.testing.assert_close(actual, expected[:, :, order], atol=1e-6, rtol=1e-6)


def test_missing_assets_do_not_affect_others_or_advance_hidden_state():
    model, x, categories, mask = setup_model()
    mask[:, 1:3, 1] = False
    mask[:, 3, :] = False  # An entirely empty timestamp must remain finite.
    changed = x.clone()
    changed[~mask] = float("nan")
    expected, _ = model(x, categories, mask)
    actual, _ = model(changed, categories, mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    _, before = model(x[:, :1], categories[:, :1], mask[:, :1])
    _, after = model(x[:, 1:3], categories[:, 1:3], mask[:, 1:3], before)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a[:, 1], b[:, 1])


def test_cached_execution_matches_whole_day_with_missing_assets():
    model, x, categories, mask = setup_model()
    mask[:, 1:3, 1] = False
    expected, _ = model(x, categories, mask)
    state, outputs = None, []
    for t in range(x.shape[1]):
        out, state = model(x[:, t : t + 1], categories[:, t : t + 1], mask[:, t : t + 1], state)
        outputs.append(out)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-6)


def test_multitask_loss_gradient_and_zero_energy_handling():
    api = model_api()
    prediction = torch.zeros(1, 1, 1, 9, requires_grad=True)
    target = torch.ones_like(prediction)
    weights = torch.ones(1, 1, 1)
    loss = api.multitask_loss(prediction, target, weights, (1,) * 9, balance=True)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, -2 / 9))
    zero = api.multitask_loss(prediction, torch.zeros_like(target), weights, (1,) * 9)
    assert zero.item() == 0 and torch.isfinite(zero)
