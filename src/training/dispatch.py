"""Select a training backend while keeping CV, API replay, and metrics shared."""

from src.training import offline, patrick


def prepare(config, source, dates, directory):
    backend = patrick if getattr(config, "method", None) == "patrick" else offline
    return backend.prepare_training(source, dates, config.features, directory)


def fit(config, prepared, model_config, seed):
    if getattr(config, "method", None) == "patrick":
        return patrick.train_model(prepared, model_config, training=config.training, seed=seed)
    return offline.train_model(
        prepared,
        model_config,
        seed=seed,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        device=config.training.device,
    )
