"""Patrick development training with frozen validation after every completed epoch."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.artifacts import save_predictor, write_json
from src.cv import configured_folds
from src.data.loader import RestrictedDateSource
from src.data.patrick_features import PatrickFeatures
from src.experiment import source_manifest
from src.models.patrick_yam import PatrickYam
from src.monitoring import training_events
from src.training.patrick import PreparedPatrick, prepare_training, train_model
from src.training.validation import PreparedValidation, prepare_validation


def _run_development_verified(
    source, config, output, gate, *, tracker=None, resume=False, cache_root=None
):
    # This command deliberately cannot turn the already-inspected initial replay
    # into an epoch-selection set. Its normalizer must be fitted again on the
    # shorter development training interval.
    if (
        getattr(config, "method", None) != "patrick"
        or len(config.members) != 1
        or config.online.enabled
        or config.cv.validation_end is None
    ):
        raise ValueError("development requires Patrick, one seed, fixed dates and online disabled")
    if config.cv.validation_end >= 1380:
        raise ValueError("development validation must end before date 1380")
    folds = configured_folds(source.dates(), config.cv)
    if len(folds) != 1:
        raise ValueError("one development fold required")
    fold = folds[0]
    identity = json.loads(
        json.dumps(
            {
                "config": config.to_dict(),
                "split": asdict(fold),
                "source": source_manifest(source),
                "gate": gate,
            },
            sort_keys=True,
        )
    )
    # Gate output contains runtime; only source identity belongs in resume matching.
    identity["gate"] = {k: v for k, v in gate.items() if k in ("passed", "code_and_tests_sha256")}
    output = Path(output)
    if resume:
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("development resume identity mismatch")
        if not (output / "training.pt").is_file():
            raise ValueError("resume requires a committed training checkpoint")
        cache = json.loads((output / "preparation.json").read_text())
        directory = Path(cache["directory"])
        features = PatrickFeatures(config.features, fold.train_dates)
        features.load_state_dict(json.loads((directory / "features.json").read_text()))
        prepared = PreparedPatrick(directory, fold.train_dates, features)
        validation = PreparedValidation(output / "validation_cache")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "identity.json", identity)
        write_json(output / "safety_gate.json", gate)
        training_source = RestrictedDateSource(source, fold.train_dates)
        if cache_root is None:
            prepared = prepare_training(
                training_source, fold.train_dates, config.features, output / "training_cache"
            )
            write_json(prepared.directory / "features.json", prepared.features.state_dict())
        else:
            from src.training.cache import parquet_fingerprint, prepare_cached

            if not hasattr(source, "path"):
                raise ValueError("persistent preparation cache requires parquet input")
            digest, _ = parquet_fingerprint(source.path)
            prepared, _ = prepare_cached(
                training_source, fold.train_dates, config.features, cache_root, digest
            )
        write_json(output / "preparation.json", {"directory": str(prepared.directory.resolve())})
        validation = prepare_validation(
            RestrictedDateSource(source, fold.validation_dates),
            fold.validation_dates,
            prepared,
            output / "validation_cache",
        )
    checkpoint = output / "training.pt"
    model_config, seed = config.members[0]
    logged = int(tracker.summary.get("last_checkpoint_step", -1)) if tracker else -1

    def observe(record):
        nonlocal logged
        if not record["checkpoint_written"]:
            return
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Epoch snapshots are immutable and use CPU-only RNG isolation; observation
        # cannot consume the training model's dropout random stream.
        if saved["epoch"] and saved["position"] == 0:
            directory = output / "epochs" / f"epoch_{saved['epoch']}"
            if not directory.exists():
                import tempfile

                directory.parent.mkdir(exist_ok=True)
                with (
                    torch.random.fork_rng(devices=[]),
                    tempfile.TemporaryDirectory(
                        dir=directory.parent, prefix=".epoch-"
                    ) as temporary,
                ):
                    frozen = PatrickYam(model_config, prepared.features.vocab_sizes)
                    frozen.load_state_dict(saved["model"])
                    staged = Path(temporary) / "checkpoint"
                    save_predictor(
                        staged, [frozen], prepared.features, prepared.scaler, config.online, [seed]
                    )
                    staged.rename(directory)
        write_json(output / "history.json", saved["history"])
        state = {**record, "latest_epoch": saved["history"][-1] if saved["history"] else None}
        write_json(output / "status.json", state)
        if tracker:
            for event in training_events(
                saved, days_per_epoch=len(prepared.dates), epochs=config.training.epochs
            ):
                if event["step"] > logged:
                    tracker.log(event["metrics"], step=event["step"])
                    logged = event["step"]
            tracker.summary["last_checkpoint_step"] = logged
        if saved["position"] == 0:
            print(json.dumps(state), flush=True)

    _, history = train_model(
        prepared,
        model_config,
        training=config.training,
        seed=seed,
        checkpoint_path=checkpoint,
        validation=validation,
        progress=observe,
    )
    # Also recover observation if interrupted after the final checkpoint commit.
    observe(
        {
            "completed_batches": len(prepared.dates) * config.training.epochs,
            "epochs_completed": config.training.epochs,
            "checkpoint_written": True,
        }
    )
    result = {
        "epochs": config.training.epochs,
        "seed": seed,
        "validation_r2": history[-1]["validation"]["r2"],
        "validation": history[-1]["validation"],
        "note": "Frozen development validation; no online updates or automatic epoch selection.",
    }
    write_json(output / "result.json", result)
    return result


def main():
    import argparse
    import os

    from src.config import load_config
    from src.data.loader import ParquetSource
    from src.safety import safety_gate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/patrick_epoch_validation.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default="cweill-self")
    parser.add_argument("--wandb-id")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    config = load_config(args.config)
    gate = safety_gate()
    tracker = None
    if args.wandb_project:
        import wandb

        tracker = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_id or Path(args.output).name,
            resume="allow",
            job_type="patrick-epoch-validation",
            config=config.to_dict(),
        )
        tracker.define_metric("train/*", step_metric="train/batch")
        for name in ("mean_optimization_loss", "mean_unbalanced_loss", "responder_6_r2"):
            tracker.define_metric(f"train/epoch_{name}", step_metric="train/epoch")
        tracker.define_metric("val/*", step_metric="val/epoch")
    try:
        result = _run_development_verified(
            ParquetSource(args.data),
            config,
            args.output,
            gate,
            tracker=tracker,
            resume=args.resume,
            cache_root=args.cache,
        )
        print(json.dumps(result, indent=2))
    except BaseException:
        if tracker:
            tracker.finish(exit_code=1)
        raise
    else:
        if tracker:
            tracker.finish()


if __name__ == "__main__":
    main()
