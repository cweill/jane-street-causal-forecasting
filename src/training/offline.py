"""Train-only preparation and deterministic daily-sequence optimization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.features import FeaturePipeline, TrainOnlyStandardizer
from src.data.schema import KEYS, RESPONDERS, test_view
from src.models.grigoreva_gru import GrigorevaGRU, weighted_r2_loss


def auxiliary_targets(responders: np.ndarray):
    """Build [r10, r9, r8, r7] within an already isolated training partition.

    Forward shifts refer to observations of ONE symbol, not calendar days.
    Missing future endpoints are masked, never borrowed from validation.
    """
    n = len(responders)
    values, mask = np.zeros((n, 4), dtype=np.float32), np.zeros((n, 4), dtype=bool)
    values[:, 2:] = responders[:, [8, 7]]
    mask[:, 2:] = np.isfinite(values[:, 2:])
    if n > 4:
        values[:-4, 1] = responders[:-4, 8] + responders[4:, 8]
        mask[:-4, 1] = np.isfinite(values[:-4, 1])
    if n > 40:
        values[:-40, 0] = responders[:-40, 6] + responders[20:-20, 6] + responders[40:, 6]
        mask[:-40, 0] = np.isfinite(values[:-40, 0])
    values[~mask] = 0.0
    return values, mask


def pack_day(x, keys, target, weights, auxiliary=None, auxiliary_mask=None):
    """One batch = one day; each symbol has its own observed-time sequence.

    Right padding has zero weight. Missing timestamps do not invent observations.
    Inputs/labels retain chronological order within each symbol.
    """
    keys = np.asarray(keys)
    if len(np.unique(keys[:, 0])) != 1 or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("pack_day requires unique keys from one date")
    groups = []
    for symbol in np.unique(keys[:, 2]):
        indices = np.flatnonzero(keys[:, 2] == symbol)
        groups.append(indices[np.argsort(keys[indices, 1], kind="stable")])
    shape = (len(groups), max(map(len, groups)))
    xx = np.zeros((*shape, x.shape[1]), np.float32)
    yy, ww = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    aa, mm = np.zeros((*shape, 4), np.float32), np.zeros((*shape, 4), bool)
    for i, indices in enumerate(groups):
        n = len(indices)
        xx[i, :n], yy[i, :n], ww[i, :n] = x[indices], target[indices], weights[indices]
        if auxiliary is not None:
            aa[i, :n], mm[i, :n] = auxiliary[indices], auxiliary_mask[indices]
    return xx, yy, ww, aa, mm


@dataclass
class PreparedTraining:
    directory: Path
    dates: tuple[int, ...]
    features: FeaturePipeline
    scaler: TrainOnlyStandardizer

    def read(self, date):
        if date not in self.dates:
            raise ValueError("date outside training partition")
        with np.load(self.directory / f"{date}.npz", allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}


def prepare_training(source, training_dates, feature_config, directory):
    dates = tuple(training_dates)
    if not dates or dates != tuple(sorted(set(dates))) or not set(dates).issubset(source.dates()):
        raise ValueError("invalid training dates")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    features = FeaturePipeline(feature_config)
    scaler = TrainOnlyStandardizer(dates, len(features.names), time_index=len(features.names) - 1)
    for date in dates:
        day = source.day(date)
        batches = test_view(day).partition_by("time_id", maintain_order=True)
        x = np.concatenate([features.transform(batch) for batch in batches])
        scaler.update(x, date=date)
        np.savez(
            directory / f"{date}.npz",
            x=x.astype(np.float32),
            keys=day.select(KEYS).to_numpy(),
            responders=day.select(RESPONDERS).to_numpy(),
            weight=day["weight"].to_numpy(),
        )
    scaler.freeze()
    prepared = PreparedTraining(directory, dates, features, scaler)
    # Reverse traversal needs only 40 subsequent observations per symbol in RAM.
    # The source is never queried outside declared training dates.
    future = {}
    for date in reversed(dates):
        data = prepared.read(date)
        aux = np.zeros((len(data["keys"]), 4), np.float32)
        mask = np.zeros_like(aux, dtype=bool)
        for symbol in np.unique(data["keys"][:, 2]):
            indices = np.flatnonzero(data["keys"][:, 2] == symbol)
            sequence = np.concatenate(
                [data["responders"][indices], future.get(symbol, np.empty((0, 9)))]
            )
            a, m = auxiliary_targets(sequence)
            aux[indices], mask[indices] = a[: len(indices)], m[: len(indices)]
            future[symbol] = sequence[:40].copy()
        np.savez(directory / f"{date}.npz", **data, auxiliary=aux, auxiliary_mask=mask)
    return prepared


def train_model(prepared, model_config, *, seed=0, epochs=1, learning_rate=0.0005, device="cpu"):
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = GrigorevaGRU(len(prepared.features.names), model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for date in rng.permutation(prepared.dates):
            data = prepared.read(int(date))
            arrays = pack_day(
                prepared.scaler.transform(data["x"]),
                data["keys"],
                data["responders"][:, 6],
                data["weight"],
                data["auxiliary"],
                data["auxiliary_mask"],
            )
            x, y, w, aux, mask = [torch.as_tensor(a, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary, _ = model(x)
            loss = weighted_r2_loss(prediction, y, w)
            if auxiliary is not None:
                for target in range(4):
                    loss = loss + weighted_r2_loss(
                        auxiliary[..., target], aux[..., target], w * mask[..., target]
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch + 1, "mean_daily_loss": float(np.mean(losses))})
    return model.eval(), history
