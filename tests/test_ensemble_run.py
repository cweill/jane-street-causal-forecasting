import copy
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_patrick_pipeline import panel, tiny_config, train

from src.artifacts import load_predictor, save_predictor, write_json
from src.data.loader import FrameSource


def test_verified_legacy_cache_rejects_changed_preparation_and_corruption(tmp_path):
    from src.ensemble_run import load_verified_cache
    from src.training.cache import ROOT, prepare_cached

    cfg = tiny_config()
    prepared, info = prepare_cached(
        FrameSource(panel()), (0, 1, 2), cfg.features, tmp_path / "cache", "a" * 64
    )
    archive = tmp_path / "source.tar.gz"
    names = [
        "src/data/schema.py",
        "src/data/features.py",
        "src/data/patrick_features.py",
        "src/training/patrick.py",
        "src/training/cache.py",
    ]
    with tarfile.open(archive, "w:gz") as target:
        for name in names:
            target.add(ROOT / name, arcname=name)
    actual = load_verified_cache(Path(info["directory"]), archive, cfg, (0, 1, 2), "a" * 64)
    assert actual.features.state_dict() == prepared.features.state_dict()
    with pytest.raises(ValueError, match="identity"):
        load_verified_cache(Path(info["directory"]), archive, cfg, (0, 1, 2, 3), "a" * 64)
    changed = tmp_path / "changed.tar.gz"
    with tarfile.open(changed, "w:gz") as target:
        for name in names:
            data = (ROOT / name).read_bytes()
            if name == "src/training/patrick.py":
                data = data.replace(
                    b"public = public.sort(KEYS)", b"public = public.sort(KEYS).head(1)"
                )
            item = tarfile.TarInfo(name)
            item.size = len(data)
            target.addfile(item, io.BytesIO(data))
    with pytest.raises(ValueError, match="preparation"):
        load_verified_cache(Path(info["directory"]), changed, cfg, (0, 1, 2), "a" * 64)
    (prepared.directory / "0.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_verified_cache(Path(info["directory"]), archive, cfg, (0, 1, 2), "a" * 64)


def test_assembly_requires_complete_distinct_seeds_and_identical_preprocessing(tmp_path):
    from src.ensemble_run import assemble_ensemble, config_fingerprint

    _, prepared, cfg = train(tmp_path)
    cfg = replace(cfg, ensemble=replace(cfg.ensemble, seed_ensembling=True, seeds=(0, 1)))
    base = load_predictor(tmp_path / "model")
    roots = [tmp_path / f"seed_{i}" for i in (0, 1)]
    for i, root in enumerate(roots):
        model = copy.deepcopy(base.models[0])
        if i:
            with torch.no_grad():
                model.head[-1].bias.add_(0.1)
        save_predictor(
            root / "checkpoint", [model], prepared.features, prepared.scaler, cfg.online, [i]
        )
        write_json(
            root / "result.json",
            {
                "seed": i,
                "status": "complete",
                "epochs": cfg.training.epochs,
                "config_sha256": config_fingerprint(cfg),
            },
        )
    assemble_ensemble(roots, cfg, tmp_path / "ensemble")
    predictor = load_predictor(tmp_path / "ensemble")
    assert predictor.seeds == (0, 1) and predictor.stacked_inference
    with pytest.raises(ValueError, match="seed"):
        assemble_ensemble([roots[0], roots[0]], cfg, tmp_path / "duplicate")
    record = json.loads((roots[1] / "result.json").read_text())
    record["epochs"] = 0
    write_json(roots[1] / "result.json", record)
    with pytest.raises(ValueError, match="complete"):
        assemble_ensemble(roots, cfg, tmp_path / "unfinished")
    record["epochs"] = cfg.training.epochs
    write_json(roots[1] / "result.json", record)
    metadata_path = roots[1] / "checkpoint/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["feature_state"]["scaler"]["mean"][0] += 1
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="preprocessing"):
        assemble_ensemble(roots, cfg, tmp_path / "different_scaler")


def test_frozen_replay_checks_every_member(tmp_path, monkeypatch):
    from src.parallel_replay import run_replay_mode
    from src.training.patrick import PatrickPredictor

    source, prepared, cfg = train(tmp_path)
    base = load_predictor(tmp_path / "model")
    save_predictor(
        tmp_path / "ensemble",
        [base.models[0], copy.deepcopy(base.models[0])],
        prepared.features,
        prepared.scaler,
        cfg.online,
        [0, 1],
        stacked_inference=True,
    )
    original = PatrickPredictor.predict

    def bad(self, test, lags):
        result = original(self, test, lags)
        if not self.config.enabled:
            with torch.no_grad():
                self.models[1].head[-1].bias.add_(0.1)
        return result

    monkeypatch.setattr(PatrickPredictor, "predict", bad)
    with pytest.raises(ValueError, match="frozen replay mutated"):
        run_replay_mode(
            source,
            tmp_path / "replay",
            checkpoint=tmp_path / "ensemble",
            mode="offline",
            replay_dates=(3, 4),
            scored_dates=(4,),
            device="cpu",
            provenance={},
        )
