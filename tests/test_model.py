import numpy as np
import pytest
import torch

from src.models.grigoreva_gru import GrigorevaGRU, ModelConfig, weighted_r2_loss
from src.training.offline import auxiliary_targets, pack_day


@pytest.mark.parametrize("architecture", ["gru3", "gru_mlp"])
def test_published_architectures_and_auxiliary_branch_wiring(architecture):
    model = GrigorevaGRU(125, ModelConfig(architecture=architecture, auxiliary_targets=True))
    assert len(model.branches) == 4
    branch = model.branches[0]
    sizes = [layer.hidden_size for layer in branch.grus]
    assert sizes == ([250, 150, 150] if architecture == "gru3" else [500])
    assert model.combiner.in_features == 4
    assert sum(p.numel() for p in model.parameters()) > 1_000_000
    with torch.no_grad():
        prediction, auxiliary, _ = model.eval()(torch.zeros(2, 3, 125))
    assert prediction.shape == (2, 3)
    assert auxiliary.shape == (2, 3, 4)


def test_sequence_and_incremental_forward_agree_and_future_x_cannot_affect_past():
    torch.manual_seed(2)
    model = GrigorevaGRU(
        3, ModelConfig(hidden_sizes=(7,), linear_sizes=(5,), dropout=(0.0,), linear_dropout=(0.0,))
    ).eval()
    x = torch.randn(2, 6, 3)
    full, _, _ = model(x)
    hidden, steps = None, []
    for t in range(6):
        p, _, hidden = model(x[:, t : t + 1], hidden)
        steps.append(p)
    torch.testing.assert_close(torch.cat(steps, dim=1), full)
    changed = x.clone()
    changed[:, 3:] *= 10000
    modified, _, _ = model(changed)
    torch.testing.assert_close(modified[:, :3], full[:, :3])


def test_auxiliary_shifts_are_observations_per_symbol_and_censored_at_boundary():
    responders = np.zeros((50, 9))
    responders[:, 6] = np.arange(50) + 1
    responders[:, 8] = np.arange(50) + 100
    aux, mask = auxiliary_targets(responders)
    assert aux[0, 0] == 1 + 21 + 41  # responder_10
    assert aux[0, 1] == 100 + 104  # responder_9
    assert mask[9, 0] and not mask[10, 0]
    assert mask[45, 1] and not mask[46, 1]
    assert np.all(aux[~mask] == 0)


def test_ragged_day_packing_never_mixes_symbols_or_scores_padding():
    keys = np.array([[1, 0, 9], [1, 0, 2], [1, 1, 2], [1, 2, 9], [1, 3, 44]])
    packed = pack_day(
        np.arange(5, dtype=float)[:, None], keys, np.arange(5, dtype=float), np.ones(5)
    )
    x, y, weights, _, _ = packed
    np.testing.assert_array_equal(x[:, :, 0], [[1, 2], [0, 3], [4, 0]])
    assert weights[2, 1] == 0
    assert y[1, 1] == 3


def test_loss_uses_mask_before_arithmetic_and_rejects_bad_shapes():
    pred = torch.tensor([1.0, 100.0], requires_grad=True)
    loss = weighted_r2_loss(pred, torch.tensor([2.0, float("nan")]), torch.tensor([1.0, 0.0]))
    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert pred.grad[1] == 0
    with pytest.raises(ValueError):
        weighted_r2_loss(pred, torch.ones(2, 1), torch.ones(2))
