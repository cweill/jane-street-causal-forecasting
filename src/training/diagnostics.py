"""Detached training diagnostics from the existing pre-update forward pass.

These are in-sample, train-mode observations (including dropout), not validation
scores or a fixed-checkpoint evaluation. Date multipliers are deliberately excluded.
"""

import math

import torch


@torch.no_grad()
def batch_diagnostics(prediction, target, weight, target_weights):
    # The optimization loss validates these inputs before this function is called.
    active = weight > 0
    p, y, w = prediction[active].double(), target[active].double(), weight[active].double()
    sse = (w[:, None] * (p - y).square()).sum(dim=0)
    energy = (w[:, None] * y.square()).sum(dim=0)
    importance = energy.new_tensor(target_weights) * (energy > 0)
    total = importance.sum()
    raw = (sse / energy.clamp_min(1e-12) * importance).sum() / total.clamp_min(1e-12)
    raw, total, sse6, energy6 = torch.stack((raw, total, sse[6], energy[6])).cpu().tolist()
    return {
        "unbalanced_loss": raw if total > 0 else None,
        "responder_6_sse": sse6,
        "responder_6_energy": energy6,
    }


def epoch_diagnostics(records):
    losses = [r["unbalanced_loss"] for r in records if r["unbalanced_loss"] is not None]
    sse = math.fsum(r["responder_6_sse"] for r in records)
    energy = math.fsum(r["responder_6_energy"] for r in records)
    return {
        "mean_unbalanced_loss": math.fsum(losses) / len(losses) if losses else None,
        "responder_6_r2": 1 - sse / energy if energy > 0 else None,
    }
