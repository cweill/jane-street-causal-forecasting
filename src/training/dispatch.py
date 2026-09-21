"""Adapters for Patrick preparation and training in the experiment runner."""

from src.training import patrick


def prepare(config, source, dates, directory):
    return patrick.prepare_training(source, dates, config.features, directory)


def fit(config, prepared, model_config, seed):
    return patrick.train_model(prepared, model_config, training=config.training, seed=seed)
