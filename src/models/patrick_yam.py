"""Documented reconstruction of Patrick Yam's axial attention / temporal GRU model."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatrickModelConfig:
    d_model: int = 64
    nheads: int = 8
    d_hidden: int = 1024
    layers: int = 8
    rnn_multiplier: int = 4
    d_cat: int = 16
    dropout: float = 0.0
    head_sizes: tuple[int, ...] = (256, 128)
    post_norm: bool = False
    auxiliary_targets: bool = True
    balance_losses: bool = True
    target_weights: tuple[float, ...] = (1, 1, 1, 6, 2, 2, 12, 5, 5)

    def __post_init__(self):
        if (
            min(
                self.d_model,
                self.nheads,
                self.d_hidden,
                self.layers,
                self.rnn_multiplier,
                self.d_cat,
            )
            < 1
        ):
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.nheads or self.d_hidden % 2 or not 0 <= self.dropout < 1:
            raise ValueError("invalid attention/gated width or dropout")
        if (
            len(self.target_weights) != 9
            or min(self.target_weights) < 0
            or self.target_weights[6] <= 0
        ):
            raise ValueError("nine nonnegative target weights with positive responder_6 required")


class SiLUGate(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * nn.functional.silu(gate)


class AxialGRUBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.norm1, self.norm2, self.norm3 = [nn.RMSNorm(d, eps=1e-5) for _ in range(3)]
        self.attention = nn.MultiheadAttention(
            d, config.nheads, dropout=config.dropout, batch_first=True
        )
        self.gru = nn.GRU(d, d * config.rnn_multiplier, batch_first=True)
        self.projection = nn.Linear(d * config.rnn_multiplier, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_hidden),
            SiLUGate(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden // 2, d),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, mask, state=None):
        b, t, a, d = x.shape
        z = self.norm1(x).reshape(b * t, a, d)
        padding = ~mask.reshape(b * t, a)
        # Attention over an empty timestamp otherwise produces NaNs. A dummy zero
        # key is visible internally; all its outputs remain masked out.
        padding = padding.clone()
        padding[:, 0] &= ~padding.all(dim=1)
        z = self.attention(z, z, z, key_padding_mask=padding, need_weights=False)[0]
        x = torch.where(mask[..., None], x + z.reshape(b, t, a, d), 0.0)
        z = self.norm2(x).permute(0, 2, 1, 3).reshape(b * a, t, d)
        active = mask.permute(0, 2, 1).reshape(b * a, t)
        if bool(active.all()):
            z, state = self.gru(z, state)
        else:
            state = z.new_zeros(1, b * a, self.gru.hidden_size) if state is None else state
            pieces = []
            for index in range(t):
                _, candidate = self.gru(z[:, index : index + 1], state)
                state = torch.where(active[:, index][None, :, None], candidate, state)
                pieces.append(state[0])
            z = torch.stack(pieces, dim=1)
        z = self.projection(z).reshape(b, a, t, d).permute(0, 2, 1, 3)
        x = torch.where(mask[..., None], x + z, 0.0)
        return torch.where(mask[..., None], x + self.ffn(self.norm3(x)), 0.0), state


class PatrickYam(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.config = config
        self.vocab_sizes = tuple(vocab_sizes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, config.d_cat, padding_idx=0) for n in vocab_sizes]
        )
        self.input = nn.Linear(77 + 3 * config.d_cat, config.d_model)
        self.blocks = nn.ModuleList([AxialGRUBlock(config) for _ in range(config.layers)])
        self.post_norm = nn.RMSNorm(config.d_model, eps=1e-5) if config.post_norm else nn.Identity()
        head, width = [], config.d_model
        for next_width in config.head_sizes:
            head.extend([nn.Linear(width, next_width), nn.SiLU()])
            width = next_width
        head.append(nn.Linear(width, 9))
        self.head = nn.Sequential(*head)

    def forward(self, x, categories, mask, state=None):
        if (
            x.ndim != 4
            or x.shape[-1] != 77
            or categories.shape != (*x.shape[:-1], 3)
            or mask.shape != x.shape[:-1]
        ):
            raise ValueError("expected numerical B,T,A,77, categorical B,T,A,3 and presence B,T,A")
        x = torch.where(mask[..., None], x, 0.0).clamp(-10, 10)
        categories = torch.where(mask[..., None], categories, 0)
        x = self.input(
            torch.cat(
                [
                    x,
                    *[embedding(categories[..., i]) for i, embedding in enumerate(self.embeddings)],
                ],
                dim=-1,
            )
        )
        x = torch.where(mask[..., None], x, 0.0)
        old_states = [None] * len(self.blocks) if state is None else state
        states = []
        for block, old in zip(self.blocks, old_states, strict=True):
            x, new = block(x, mask, old)
            states.append(new)
        return torch.where(mask[..., None], self.head(self.post_norm(x)), 0.0), states


def multitask_loss(prediction, target, weight, target_weights, *, balance=True):
    if prediction.shape != target.shape or prediction.shape[:-1] != weight.shape:
        raise ValueError("incompatible multitask shapes")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    active = weight > 0
    if not torch.isfinite(prediction[active]).all() or not torch.isfinite(target[active]).all():
        raise ValueError("nonfinite active targets/predictions")
    p, y, w = prediction[active], target[active], weight[active, None]
    energy = (w * y.square()).sum(dim=0)
    sse = (w * (p - y).square()).sum(dim=0)
    importance = prediction.new_tensor(target_weights) * (energy > 0)
    ratios = sse / energy.clamp_min(1e-12)
    if balance:
        ratios = ratios / ratios.detach().clamp_min(1e-8)
    return (ratios * importance).sum() / importance.sum().clamp_min(1e-12)
