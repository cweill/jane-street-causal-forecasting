import importlib
import importlib.util
import json
from dataclasses import replace

import polars as pl
import pytest
from test_patrick_pipeline import panel, tiny_config

from src.artifacts import load_predictor, save_predictor, sha256_file
from src.cv import TemporalFold
from src.data.loader import FrameSource
from src.training.patrick import prepare_training, train_model


def api():
    assert importlib.util.find_spec("src.online_sweep"), "controlled online sweep missing"
    return importlib.import_module("src.online_sweep")


def fixture(tmp_path):
    config, source = tiny_config(), FrameSource(panel())
    prepared = prepare_training(source, (0, 1, 2), config.features, tmp_path / "train")
    models = [
        train_model(prepared, config.model, training=config.training, seed=s)[0] for s in (0, 1)
    ]
    base = tmp_path / "initial"
    save_predictor(
        base,
        models,
        prepared.features,
        prepared.scaler,
        config.online,
        [0, 1],
        stacked_inference=True,
    )
    return source, base, TemporalFold(0, (0, 1, 2), (3,), (4, 5, 6))


def test_trials_change_only_online_settings_and_share_frozen_initial_predictions(tmp_path):
    module = api()
    source, base, fold = fixture(tmp_path)
    initial = {p.name: p.read_bytes() for p in base.iterdir()}
    trials = [
        module.Trial("frozen", None, False),
        module.Trial("persistent", 1e-4, False),
        module.Trial("reset", 1e-4, True),
    ]
    results = []
    for trial in trials:
        root = tmp_path / "trials" / trial.name
        result = module.run_trial(
            source, base, root, trial, fold, device="cpu", provenance={"test": "fixed"}
        )
        results.append(result)
        assert result["initial_weights_sha256"] == results[0]["initial_weights_sha256"]
        assert result["denominator"] == results[0]["denominator"]
        assert sha256_file(root / "initial_checkpoint/weights.pt") == sha256_file(
            base / "weights.pt"
        )
        mode = "offline" if trial.learning_rate is None else "online"
        first = pl.read_parquet(root / mode / "date_3.parquet")
        assert first.equals(pl.read_parquet(tmp_path / "trials/frozen/offline/date_3.parquet"))
        loaded = load_predictor(root / "initial_checkpoint")
        if trial.learning_rate is not None:
            assert loaded.config.learning_rate == trial.learning_rate
            assert loaded.config.reset_daily_optimizer == trial.reset_daily
            assert len(result["updates"]) == 3
            assert all(u["optimizer_reset_daily"] == trial.reset_daily for u in result["updates"])
            assert all(u["source_date"] == u["released_at"][0] - 1 for u in result["updates"])
        assert loaded.config.steps == 3
        assert tuple(loaded.config.betas) == (0.8, 0.95)
    assert results[0]["final_weights_sha256"] == results[0]["initial_weights_sha256"]
    assert results[1]["final_weights_sha256"] != results[2]["final_weights_sha256"]
    assert initial == {p.name: p.read_bytes() for p in base.iterdir()}
    summary = module.summarize_trials(tmp_path / "trials", trials, fold, window=2)
    assert len(summary["trials"]) == 3
    assert summary["trials"][1]["delta_r2"] == results[1]["score"] - results[0]["score"]
    assert (tmp_path / "trials/rolling_comparison.png").exists()
    identity_path = tmp_path / "trials/reset/online/identity.json"
    identity_text = identity_path.read_text()
    identity = json.loads(identity_text)
    identity["provenance"]["test"] = "different data"
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="paired"):
        module.summarize_trials(tmp_path / "trials", trials, fold, window=2)
    identity_path.write_text(identity_text)
    path = tmp_path / "trials/reset/result.json"
    bad = json.loads(path.read_text())
    bad["denominator"] += 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="paired"):
        module.summarize_trials(tmp_path / "trials", trials, fold, window=2)


def test_trial_resume_and_future_responder_poisoning(tmp_path):
    module = api()
    source, base, fold = fixture(tmp_path)
    trial = module.Trial("reset", 5e-4, True)

    def interrupt(record):
        if record.get("completed_date") == 4:
            raise InterruptedError()

    with pytest.raises(InterruptedError):
        module.run_trial(
            source,
            base,
            tmp_path / "resumed",
            trial,
            fold,
            device="cpu",
            provenance={},
            progress=interrupt,
        )
    actual = module.run_trial(
        source, base, tmp_path / "resumed", trial, fold, device="cpu", provenance={}
    )
    poisoned = FrameSource(
        panel().with_columns(
            [
                pl.when(pl.col("date_id") == 6)
                .then(pl.lit(999.0))
                .otherwise(pl.col(f"responder_{i}"))
                .alias(f"responder_{i}")
                for i in range(9)
            ]
        )
    )
    altered = module.run_trial(
        poisoned, base, tmp_path / "poisoned", trial, fold, device="cpu", provenance={}
    )
    for date in fold.replay_dates:
        assert pl.read_parquet(tmp_path / f"resumed/online/date_{date}.parquet").equals(
            pl.read_parquet(tmp_path / f"poisoned/online/date_{date}.parquet")
        )
    assert actual["final_weights_sha256"] == altered["final_weights_sha256"]
    assert actual["score"] != altered["score"]
    with pytest.raises(ValueError, match="identity"):
        module.run_trial(
            source,
            base,
            tmp_path / "resumed",
            replace(trial, learning_rate=1e-3),
            fold,
            device="cpu",
            provenance={},
        )


def test_rejects_leaked_initial_training_boundary_and_current_replay(tmp_path):
    module = api()
    _source, base, fold = fixture(tmp_path)
    trial = module.Trial("frozen", None, False)

    class NoReads:
        def dates(self):
            return tuple(range(1699))

        def day(self, date):
            raise AssertionError("unsafe fold read rows")

    for invalid in [
        replace(fold, train_dates=(0, 1)),
        TemporalFold(0, (0, 1, 2), tuple(range(3, 1180)), tuple(range(1180, 1381))),
    ]:
        with pytest.raises(ValueError, match="training|1380"):
            module.run_trial(
                NoReads(), base, tmp_path / "bad", trial, invalid, device="cpu", provenance={}
            )


def test_replay_day_cache_is_restricted_and_detects_corruption(tmp_path):
    module = api()
    source = FrameSource(panel())
    cache = module.prepare_replay_days(source, (2, 3, 4), tmp_path / "days")
    assert cache.dates() == (2, 3, 4)
    assert cache.day(3).equals(source.day(3))
    with pytest.raises(ValueError, match="partition"):
        cache.day(5)
    changed = FrameSource(panel().with_columns(pl.col("responder_6") + 1))
    with pytest.raises(ValueError, match="source"):
        module.prepare_replay_days(changed, (2, 3, 4), tmp_path / "days")
    with (tmp_path / "days/3.parquet").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        cache.day(3)
