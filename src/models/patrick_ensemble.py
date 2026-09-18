"""Inference-only Patrick ensemble with a leading model axis and stacked weights.

Dense and GRU projections use einsum; attention batches over models and heads.
Original modules remain the owners of trainable weights and optimizer states.
Refresh this snapshot after updates. No ensemble-member loop runs during forward.
"""

import json
from dataclasses import asdict

import torch
from torch.nn import functional as F


class StackedPatrick:
    def __init__(self, models):
        self.refresh(models)

    @torch.no_grad()
    def refresh(self, models):
        if not models:
            raise ValueError("nonempty homogeneous Patrick ensemble required")
        config = json.dumps(asdict(models[0].config), sort_keys=True)
        parameters = [next(model.parameters()) for model in models]
        first = parameters[0]
        if any(
            json.dumps(asdict(model.config), sort_keys=True) != config
            or model.vocab_sizes != models[0].vocab_sizes
            or parameter.device != first.device
            or parameter.dtype != first.dtype
            for model, parameter in zip(models, parameters, strict=True)
        ):
            raise ValueError("homogeneous architecture, vocabularies, device and dtype required")
        self.config, self.size = models[0].config, len(models)
        snapshots = [model.state_dict() for model in models]
        self.weights = {
            key: torch.stack([s[key].detach() for s in snapshots]) for key in snapshots[0]
        }

    def _linear(self, x, name):
        return (
            torch.einsum("nad,nod->nao", x, self.weights[name + ".weight"])
            + self.weights[name + ".bias"][:, None]
        )

    def _norm(self, x, name):
        return F.rms_norm(x, (x.shape[-1],), eps=1e-5) * self.weights[name + ".weight"][:, None]

    @torch.no_grad()
    def forward_step(self, numeric, categories, states=None):
        if (
            numeric.ndim != 2
            or numeric.shape[-1] != 77
            or categories.shape != (numeric.shape[0], 3)
        ):
            raise ValueError("expected one timestamp: A,77 numerical and A,3 categorical")
        n, a, d = self.size, numeric.shape[0], self.config.d_model
        if not a:
            raise ValueError("nonempty timestamp required")
        shape = (n, self.config.layers, a, d * self.config.rnn_multiplier)
        if states is None:
            states = numeric.new_zeros(shape)
        elif states.shape != shape:
            raise ValueError("incompatible stacked hidden state")
        x = torch.cat(
            [
                numeric.clamp(-10, 10).unsqueeze(0).expand(n, -1, -1),
                *[self.weights[f"embeddings.{i}.weight"][:, categories[:, i]] for i in range(3)],
            ],
            dim=-1,
        )
        x = self._linear(x, "input")
        new_states = []
        heads = self.config.nheads
        for index in range(self.config.layers):
            prefix = f"blocks.{index}"
            z = self._norm(x, prefix + ".norm1")
            qkv = torch.einsum(
                "nad,nod->nao", z, self.weights[prefix + ".attention.in_proj_weight"]
            )
            qkv = qkv + self.weights[prefix + ".attention.in_proj_bias"][:, None]
            q, k, v = qkv.reshape(n, a, 3, heads, d // heads).unbind(dim=2)
            z = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
            )
            z = z.transpose(1, 2).reshape(n, a, d)
            x = x + self._linear(z, prefix + ".attention.out_proj")
            z = self._norm(x, prefix + ".norm2")
            previous = states[:, index]
            gi = (
                torch.einsum("nad,nod->nao", z, self.weights[prefix + ".gru.weight_ih_l0"])
                + self.weights[prefix + ".gru.bias_ih_l0"][:, None]
            )
            gh = (
                torch.einsum("nad,nod->nao", previous, self.weights[prefix + ".gru.weight_hh_l0"])
                + self.weights[prefix + ".gru.bias_hh_l0"][:, None]
            )
            ir, iz, candidate = gi.chunk(3, dim=-1)
            hr, hz, hn = gh.chunk(3, dim=-1)
            reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            candidate = torch.tanh(candidate + reset * hn)
            hidden = candidate + update * (previous - candidate)
            new_states.append(hidden)
            x = x + self._linear(hidden, prefix + ".projection")
            value, gate = self._linear(self._norm(x, prefix + ".norm3"), prefix + ".ffn.0").chunk(
                2, dim=-1
            )
            x = x + self._linear(value * F.silu(gate), prefix + ".ffn.3")
        if self.config.post_norm:
            x = self._norm(x, "post_norm")
        for i in range(len(self.config.head_sizes) + 1):
            x = self._linear(x, f"head.{2 * i}")
            if i < len(self.config.head_sizes):
                x = F.silu(x)
        return x, torch.stack(new_states, dim=1)
