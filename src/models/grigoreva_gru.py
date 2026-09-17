"""Independent implementation of the published Grigoreva recurrent architectures."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "gru_mlp"
    auxiliary_targets: bool = False
    # Explicit overrides support small correctness tests without changing published presets.
    hidden_sizes: tuple[int, ...] | None = None
    linear_sizes: tuple[int, ...] | None = None
    dropout: tuple[float, ...] | None = None
    linear_dropout: tuple[float, ...] | None = None

    def dimensions(self):
        if self.architecture not in {"gru3", "gru_mlp"}:
            raise ValueError(f"unknown architecture: {self.architecture}")
        h, d, linear, ld = (
            ((250, 150, 150), (0.0, 0.0, 0.0), (), ())
            if self.architecture == "gru3"
            else ((500,), (0.3,), (500, 300), (0.2, 0.1))
        )
        h = h if self.hidden_sizes is None else tuple(self.hidden_sizes)
        linear = linear if self.linear_sizes is None else tuple(self.linear_sizes)
        d = d if self.dropout is None else tuple(self.dropout)
        ld = ld if self.linear_dropout is None else tuple(self.linear_dropout)
        if not h or len(h) != len(d) or len(linear) != len(ld):
            raise ValueError("each GRU/linear layer needs a dropout rate")
        if any(n < 1 for n in h + linear) or any(not 0 <= p < 1 for p in d + ld):
            raise ValueError("invalid model dimensions/dropout")
        return h, d, linear, ld


class RecurrentBranch(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        hidden, dropout, linear, linear_dropout = config.dimensions()
        self.grus, self.dropouts = nn.ModuleList(), nn.ModuleList()
        for size, rate in zip(hidden, dropout, strict=True):
            self.grus.append(nn.GRU(input_size, size, batch_first=True))
            self.dropouts.append(nn.Dropout(rate))
            input_size = size
        layers = []
        for size, rate in zip(linear, linear_dropout, strict=True):
            layers.extend([nn.Linear(input_size, size), nn.ReLU(), nn.Dropout(rate)])
            input_size = size
        layers.append(nn.Linear(input_size, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x, hidden=None):
        hidden = [None] * len(self.grus) if hidden is None else hidden
        state = []
        for gru, dropout, h in zip(self.grus, self.dropouts, hidden, strict=True):
            x, h = gru(x, h)
            x = dropout(x)
            state.append(h)
        return self.head(x).squeeze(-1), state


class GrigorevaGRU(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList(
            [
                RecurrentBranch(input_size, config)
                for _ in range(4 if config.auxiliary_targets else 1)
            ]
        )
        self.combiner = nn.Linear(4, 1) if config.auxiliary_targets else None

    def forward(self, x, hidden=None):
        hidden = [None] * len(self.branches) if hidden is None else hidden
        results = [b(x, h) for b, h in zip(self.branches, hidden, strict=True)]
        predictions = torch.stack([r[0] for r in results], dim=-1)
        state = [r[1] for r in results]
        if self.combiner is None:
            return predictions[..., 0], None, state
        # Branch order follows the reference loss: responder_10, _9, _8, _7.
        return self.combiner(predictions).squeeze(-1), predictions, state


def weighted_r2_loss(prediction, target, weight):
    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("loss shapes must match exactly")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("loss weights must be finite and nonnegative")
    valid = weight > 0
    p, y, w = prediction[valid], target[valid], weight[valid]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite active loss values")
    denominator = (w * y.square()).sum()
    if denominator <= 0:
        # A target with no energy provides no defined R² loss; skip this term.
        return prediction.sum() * 0.0
    return (w * (p - y).square()).sum() / denominator
