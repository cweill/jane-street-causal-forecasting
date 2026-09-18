"""Short real-data CUDA interruption/resume rehearsal; no long training run."""

import gc
import json
import uuid
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import modal

from scripts.modal_pilot import ROOT, image, volume

app = modal.App("patrick-resume-rehearsal")


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=1200,
    startup_timeout=900,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    volumes={"/pilot": volume},
    include_source=False,
)
def rehearse(run_id: str):
    import polars as pl
    import torch

    from src.artifacts import load_predictor, write_json
    from src.config import load_config
    from src.data.api_simulator import APISimulator
    from src.data.loader import FrameSource
    from src.safety import safety_gate
    from src.training.checkpoints import load_replay_checkpoint, save_replay_checkpoint
    from src.training.patrick import PreparedPatrick, train_model

    gate = safety_gate()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    baseline = Path("/pilot/runs/20260917T232828Z-0b458313")
    output = Path("/pilot/runs") / run_id
    output.mkdir()
    original = load_predictor(baseline / "checkpoint")
    prepared = PreparedPatrick(
        baseline / "training_cache", tuple(range(700, 704)), original.features
    )
    config = load_config("configs/patrick.yaml")
    # Exercise CUDA RNG restoration too; this dropout is a stress test, not the plot setting.
    model_config = replace(config.model, dropout=0.1)
    training = replace(config.training, epochs=2, device="cuda")

    class Interrupted(Exception):
        pass

    def interrupt(progress):
        if progress["completed_batches"] == 3:
            raise Interrupted()

    try:
        print("CUDA: uninterrupted two-epoch reference", flush=True)
        expected, expected_history = train_model(prepared, model_config, training=training, seed=7)
        expected = expected.cpu()
        print("CUDA: interrupt after batch 3, then restore all training state", flush=True)
        checkpoint = output / "training.pt"
        try:
            train_model(
                prepared,
                model_config,
                training=training,
                seed=7,
                checkpoint_path=checkpoint,
                checkpoint_every=1,
                progress=interrupt,
            )
        except Interrupted:
            pass
        else:
            raise AssertionError("rehearsal did not interrupt")
        volume.commit()
        gc.collect()
        torch.cuda.empty_cache()
        actual, history = train_model(
            prepared,
            model_config,
            training=training,
            seed=7,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
        )
        actual = actual.cpu()
        assert history == expected_history
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
        del expected, actual
        gc.collect()
        torch.cuda.empty_cache()
        print("CUDA: online replay checkpoint after a real daily update", flush=True)
        digest = json.loads((baseline / "data_manifest.json").read_text())["sha256"]
        source = FrameSource(pl.read_parquet(Path("/pilot/inputs") / f"{digest}.parquet"))
        uninterrupted = load_predictor(baseline / "checkpoint", "cuda")
        uninterrupted.config = replace(uninterrupted.config, enabled=True, learning_rate=0.0005)
        prefix = APISimulator(source, [704, 705]).run(uninterrupted.predict).predictions
        snapshot = output / "replay_705"
        save_replay_checkpoint(
            uninterrupted, snapshot, signature="cuda-rehearsal", row_offset=prefix.height
        )
        volume.commit()
        restored, _ = load_replay_checkpoint(snapshot, signature="cuda-rehearsal", device="cuda")
        expected = (
            APISimulator(source, [706], row_offset=prefix.height)
            .run(uninterrupted.predict)
            .predictions
        )
        actual = (
            APISimulator(source, [706], row_offset=prefix.height).run(restored.predict).predictions
        )
        assert actual.equals(expected)
        assert restored.update_log == uninterrupted.update_log
        for name, value in uninterrupted.models[0].state_dict().items():
            torch.testing.assert_close(restored.models[0].state_dict()[name], value, rtol=0, atol=0)
        summary = {
            "status": "passed",
            "gpu": torch.cuda.get_device_name(),
            "training_weights_and_history_identical": True,
            "resumed_online_predictions_and_weights_identical": True,
            "online_learning_rate": 0.0005,
            "training_stress_dropout": 0.1,
            "seconds": perf_counter() - started,
            "safety_gate": gate,
        }
        write_json(output / "result.json", summary)
        return summary
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    run_id = "resume-rehearsal-" + uuid.uuid4().hex[:10]
    result = rehearse.remote(run_id)
    destination = ROOT / "artifacts/modal-pilot" / run_id
    destination.mkdir()
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifacts": str(destination), **result}, indent=2))
