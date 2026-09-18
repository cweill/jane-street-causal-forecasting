"""Daily offline fitting and strictly delayed online updates for Patrick's model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.patrick_features import PatrickFeatures
from src.data.schema import KEYS, LAG_COLUMNS, RESPONDERS, test_view, validate_test
from src.models.patrick_yam import PatrickYam, multitask_loss


@dataclass(frozen=True)
class PatrickTrainingConfig:
    epochs: int = 1
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    betas: tuple[float, float] = (0.95, 0.9999)
    device: str = "cpu"
    recency_weighting: bool = True
    full_length_weighting: bool = True
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class PatrickOnlineConfig:
    enabled: bool = False
    learning_rate: float = 0.0005  # User hypothesis; not recovered from the talk.
    steps: int = 3
    betas: tuple[float, float] = (0.8, 0.95)
    reset_daily_optimizer: bool = False
    gradient_clip: float = 1.0


def pack_panel(public, features, labels=None):
    public = public.sort(KEYS)
    numeric, categorical = features.transform_day(public)
    times, ti = np.unique(public["time_id"].to_numpy(), return_inverse=True)
    symbols, si = np.unique(public["symbol_id"].to_numpy(), return_inverse=True)
    shape = (1, len(times), len(symbols))
    x, cats = np.zeros((*shape, 77), np.float32), np.zeros((*shape, 3), np.int64)
    mask, y, w = (
        np.zeros(shape, bool),
        np.zeros((*shape, 9), np.float32),
        np.zeros(shape, np.float32),
    )
    x[0, ti, si], cats[0, ti, si], mask[0, ti, si] = numeric, categorical, True
    if labels is not None:
        aligned = public.select(*KEYS, "weight").join(
            labels.select(*KEYS, *RESPONDERS).with_columns(pl.lit(True).alias("labeled")),
            on=KEYS,
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        y[0, ti, si] = aligned.select(RESPONDERS).fill_null(0).to_numpy()
        w[0, ti, si] = aligned["weight"].to_numpy() * aligned["labeled"].fill_null(False).to_numpy()
    keys = [
        (tuple(key), int(t), int(symbol))
        for key, t, symbol in zip(public.select(KEYS).iter_rows(), ti, si, strict=True)
    ]
    return (x, cats, mask, y, w), keys


@dataclass
class PreparedPatrick:
    directory: Path
    dates: tuple[int, ...]
    features: PatrickFeatures

    @property
    def scaler(self):
        return self.features.scaler

    def read(self, date):
        if date not in self.dates:
            raise ValueError("date outside training partition")
        with np.load(self.directory / f"{date}.npz", allow_pickle=False) as data:
            return tuple(data[name] for name in ("x", "cats", "mask", "y", "w"))


def prepare_training(source, dates, feature_config, directory):
    dates = tuple(dates)
    if not dates or dates != tuple(sorted(set(dates))) or not set(dates).issubset(source.dates()):
        raise ValueError("invalid training dates")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    features = PatrickFeatures(feature_config, dates)
    for date in dates:
        features.update(source.day(date), date)
    features.freeze()
    for date in dates:
        day = source.day(date)
        arrays, _ = pack_panel(test_view(day), features, day)
        np.savez(
            directory / f"{date}.npz",
            **dict(zip(("x", "cats", "mask", "y", "w"), arrays, strict=True)),
        )
    return PreparedPatrick(directory, dates, features)


def loss_for_model(model, prediction, y, w):
    config = model.config
    target_weights = (
        config.target_weights if config.auxiliary_targets else (0, 0, 0, 0, 0, 0, 1, 0, 0)
    )
    return multitask_loss(prediction, y, w, target_weights, balance=config.balance_losses)


def train_model(prepared, model_config, *, training=None, seed=0):
    training = PatrickTrainingConfig() if training is None else training
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = PatrickYam(model_config, prepared.features.vocab_sizes).to(training.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
        betas=training.betas,
    )
    rng, history = np.random.default_rng(seed), []
    for epoch in range(training.epochs):
        losses = []
        model.train()
        for date in rng.permutation(prepared.dates):
            x, cats, mask, y, w = [
                torch.as_tensor(a, device=training.device) for a in prepared.read(int(date))
            ]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x, cats, mask)
            loss = loss_for_model(model, prediction, y, w)
            # Per-day multipliers must be outside the normalized daily loss;
            # multiplying both R² numerator/denominator would cancel them.
            importance = (
                (200 + date) / (200 + max(prepared.dates)) if training.recency_weighting else 1.0
            )
            if training.full_length_weighting and x.shape[1] == 968:
                importance *= 1.5
            loss = loss * importance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            optimizer.step()
            losses.append(loss.item())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_optimization_loss": float(np.mean(losses)),
                "note": "Detached loss balancing makes this unsuitable for score/early stopping.",
            }
        )
    return model.eval(), history


class PatrickPredictor:
    def __init__(self, models, features, config=None, seeds=None):
        self.models, self.features, self.scaler = models, features, features.scaler
        self.config = PatrickOnlineConfig() if config is None else config
        self.seeds = tuple(range(len(models))) if seeds is None else tuple(seeds)
        if (
            not models
            or len(models) != len(self.seeds)
            or len({id(m) for m in models}) != len(models)
        ):
            raise ValueError("distinct models and one seed per model required")
        self._states = [None for _ in models]
        self._symbol_slots = {}
        self._optimizers = [None for _ in models]
        self._day, self._last_key, self._cache = None, None, []
        self.update_log = []

    def _update(self, lags, date):
        if not self.config.enabled or not self._cache or lags is None or lags.is_empty():
            return
        public = pl.concat(self._cache)
        if public["date_id"].unique().to_list() != [date - 1]:
            return
        labels = lags.with_columns(
            pl.lit(date - 1).cast(lags["date_id"].dtype).alias("date_id")
        ).rename({name + "_lag_1": name for name in RESPONDERS})
        arrays, _ = pack_panel(public, self.features, labels)
        if np.sum(arrays[4] * arrays[3][..., 6] ** 2) <= 0:
            return
        for index, (model, seed) in enumerate(zip(self.models, self.seeds, strict=True)):
            device = next(model.parameters()).device
            with torch.random.fork_rng(
                devices=[device.index or 0] if device.type == "cuda" else []
            ):
                torch.manual_seed(int(seed) + 1_000_003 * date)
                if self._optimizers[index] is None or self.config.reset_daily_optimizer:
                    self._optimizers[index] = torch.optim.Adam(
                        model.parameters(), lr=self.config.learning_rate, betas=self.config.betas
                    )
                optimizer = self._optimizers[index]
                x, cats, mask, y, w = [torch.as_tensor(a, device=device) for a in arrays]
                model.train()
                for _ in range(self.config.steps):
                    optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(x, cats, mask)
                    loss = loss_for_model(model, prediction, y, w)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                model.eval()
        matched = public.select(KEYS).join(labels.select(KEYS), on=KEYS, how="inner").height
        self.update_log.append(
            {
                "released_at": [date, 0],
                "source_date": date - 1,
                "rows": matched,
                "models": len(self.models),
                "steps": self.config.steps,
                "target": "all_9" if self.models[0].config.auxiliary_targets else "responder_6",
                "optimizer_reset_daily": self.config.reset_daily_optimizer,
            }
        )

    def predict(self, test, lags):
        validate_test(test)
        original_rows = test.select("row_id")
        order = np.argsort(test["symbol_id"].to_numpy(), kind="stable")
        test = test[order]
        date, time = int(test["date_id"][0]), int(test["time_id"][0])
        if self._last_key is not None and (date, time) <= self._last_key:
            raise ValueError("predict calls must be strictly chronological")
        if lags is not None:
            if time != 0 or set(lags.columns) != set(KEYS + LAG_COLUMNS):
                raise ValueError("lags only at time zero with API schema")
            if lags.height and lags["date_id"].unique().to_list() != [date]:
                raise ValueError("lags must be stamped with release date")
            if lags.select(KEYS).is_duplicated().any():
                raise ValueError("duplicate lag keys")
        if date != self._day:
            self._update(lags, date)
            self._states, self._cache = [None for _ in self.models], []
            self._symbol_slots = {}
            self._day = date
        if self.config.enabled:
            self._cache.append(test.clone())
        x, cats = self.features._transform_rows(test)  # Already validated above.
        symbols = test["symbol_id"].to_list()
        for symbol in symbols:
            if symbol not in self._symbol_slots:
                self._symbol_slots[symbol] = len(self._symbol_slots)
        outputs = []
        for index, (model, states) in enumerate(zip(self.models, self._states, strict=True)):
            device = next(model.parameters()).device
            slots = torch.tensor([self._symbol_slots[s] for s in symbols], device=device)
            shape = (len(model.blocks), len(self._symbol_slots), model.blocks[0].gru.hidden_size)
            if states is None:
                states = torch.zeros(shape, device=device)
            elif states.shape[1] < shape[1]:
                states = torch.cat(
                    [states, states.new_zeros(shape[0], shape[1] - states.shape[1], shape[2])],
                    dim=1,
                )
            # One gather/scatter for the entire layer/symbol bank. Missing symbols
            # retain their columns; newly observed symbols receive zero state.
            hidden = [h.unsqueeze(0) for h in states.index_select(1, slots).unbind(0)]
            model.eval()
            with torch.no_grad():
                prediction, new = model(
                    torch.as_tensor(x[None, None], device=device),
                    torch.as_tensor(cats[None, None], device=device),
                    torch.ones(1, 1, len(symbols), device=device, dtype=torch.bool),
                    hidden,
                )
            states.index_copy_(1, slots, torch.cat(new, dim=0))
            self._states[index] = states
            outputs.append(prediction[0, 0, :, 6].cpu().numpy())
        self._last_key = (date, time)
        return original_rows.with_columns(
            pl.Series("responder_6", np.mean(outputs, axis=0)[np.argsort(order)])
        )
