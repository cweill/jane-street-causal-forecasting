"""Streaming inference and previous-day-only gradient updates."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from src.data.schema import KEYS, LAG_COLUMNS, validate_test
from src.models.grigoreva_gru import weighted_r2_loss
from src.training.offline import pack_day


@dataclass(frozen=True)
class OnlineConfig:
    enabled: bool = False
    learning_rate: float = 0.0003

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("online learning_rate must be positive")


class StreamingPredictor:
    def __init__(self, models, features, scaler, config=None, *, seeds=None, prediction_clip=5.0):
        if not models or len({id(m) for m in models}) != len(models):
            raise ValueError("ensemble needs distinct model instances")
        self.models, self.features, self.scaler = models, features, scaler
        self.config = OnlineConfig() if config is None else config
        self.seeds = tuple(range(len(models))) if seeds is None else tuple(seeds)
        if len(self.seeds) != len(models):
            raise ValueError("one seed is required per model")
        self.prediction_clip = prediction_clip
        self._states = [{} for _ in models]
        self._day = None
        self._last_key = None
        self._cache = []
        self.update_log = []

    @property
    def cached_dates(self):
        return {int(part[1][0, 0]) for part in self._cache}

    @property
    def hidden_dates(self):
        return set() if self._day is None else {self._day}

    def _validate_lags(self, lags, date, time):
        if lags is None:
            return
        if time != 0 or set(lags.columns) != set(KEYS + LAG_COLUMNS):
            raise ValueError("lags may only arrive at time_id=0 with the API lag schema")
        if lags.height and lags["date_id"].unique().to_list() != [date]:
            raise ValueError("lags date_id must equal the release date")
        if lags.select(KEYS).is_duplicated().any():
            raise ValueError("duplicate lags keys")

    def _update(self, lags, date):
        if not self.config.enabled or not self._cache or lags is None or lags.is_empty():
            return
        if self.cached_dates != {date - 1}:
            return  # A skipped day cannot be relabeled as the previous observed day.
        x, keys, weights = (np.concatenate([part[i] for part in self._cache]) for i in range(3))
        # The lag table is stamped with the release date. Re-key it to its source date
        # and join by all identifiers; never rely on row order or reshape alignment.
        labels = {
            (date - 1, int(t), int(s)): y
            for t, s, y in lags.select("time_id", "symbol_id", "responder_6_lag_1").iter_rows()
        }
        target = np.array([labels.get(tuple(key), np.nan) for key in keys], dtype=np.float32)
        observed = np.isfinite(target)
        weights = np.where(observed, weights, 0.0)
        target = np.where(observed, target, 0.0)
        if np.sum(weights * target**2) <= 0:
            return
        if self.scaler.time_index is not None:
            self.scaler.observe_released_times(keys[:, 1])
            index = self.scaler.time_index
            x[:, index] = (
                np.clip(keys[:, 1], *self.scaler.time_bounds) - self.scaler.mean[index]
            ) / self.scaler.scale[index]
        arrays = pack_day(x, keys, target, weights)
        for model, seed in zip(self.models, self.seeds, strict=True):
            device = next(model.parameters()).device
            # Stable per-member/day random streams make online ensembling independent
            # of model execution order, including dropout during updates.
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed) + 1_000_003 * int(date))
                model.train()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=self.config.learning_rate, weight_decay=0.01
                )
                xx, yy, ww = [torch.as_tensor(a, device=device) for a in arrays[:3]]
                optimizer.zero_grad(set_to_none=True)
                prediction, _, _ = model(xx)
                # Auxiliary forward labels are deliberately NEVER used online.
                loss = weighted_r2_loss(prediction, yy, ww)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.eval()
        self.update_log.append(
            {
                "released_at": [date, 0],
                "source_date": date - 1,
                "rows": int(observed.sum()),
                "models": len(self.models),
                "target": "responder_6",
            }
        )

    def _predict_member(self, model, states, symbols, x):
        device = next(model.parameters()).device
        hidden = []
        for branch_idx, branch in enumerate(model.branches):
            layers = []
            for layer_idx, layer in enumerate(branch.grus):
                layers.append(
                    torch.cat(
                        [
                            states[s][branch_idx][layer_idx]
                            if s in states
                            else torch.zeros(1, 1, layer.hidden_size, device=device)
                            for s in symbols
                        ],
                        dim=1,
                    )
                )
            hidden.append(layers)
        model.eval()
        with torch.no_grad():
            prediction, _, new_state = model(torch.as_tensor(x[:, None, :], device=device), hidden)
        for index, symbol in enumerate(symbols):
            states[symbol] = [
                [layer[:, index : index + 1].detach().clone() for layer in branch]
                for branch in new_state
            ]
        values = prediction[:, 0].cpu().numpy()
        return np.clip(values, -self.prediction_clip, self.prediction_clip)

    def predict(self, test: pl.DataFrame, lags: pl.DataFrame | None):
        validate_test(test)
        date, time = int(test["date_id"][0]), int(test["time_id"][0])
        if self._last_key is not None and (date, time) <= self._last_key:
            raise ValueError("predict calls must be strictly chronological")
        self._validate_lags(lags, date, time)
        if date != self._day:
            self._update(lags, date)
            self._cache = []
            self._states = [{} for _ in self.models]
            self._day = date
        x = self.scaler.transform(self.features.transform(test))
        if self.config.enabled:
            self._cache.append(
                (x.copy(), test.select(KEYS).to_numpy().copy(), test["weight"].to_numpy().copy())
            )
        symbols = test["symbol_id"].to_list()
        predictions = [
            self._predict_member(m, states, symbols, x)
            for m, states in zip(self.models, self._states, strict=True)
        ]
        self._last_key = (date, time)
        return test.select("row_id").with_columns(
            pl.Series("responder_6", np.mean(predictions, axis=0))
        )
