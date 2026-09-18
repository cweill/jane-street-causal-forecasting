import importlib
import importlib.util
import json

import pytest


def monitor():
    assert importlib.util.find_spec("src.monitoring"), "read-only metric extraction is missing"
    return importlib.import_module("src.monitoring")


def test_checkpoint_events_keep_real_batch_positions_and_exclude_weights():
    module = monitor()
    saved = {
        "completed": 5,
        "epoch": 1,
        "position": 2,
        "losses": [0.4, 0.6],
        "history": [{"epoch": 1, "mean_optimization_loss": 0.8}],
        "model": {"secret_tensor": "must not be uploaded"},
        "optimizer": {"private": True},
    }
    events = module.training_events(saved, days_per_epoch=3, epochs=2)
    assert [e["step"] for e in events] == [7, 8, 10]
    assert events[0]["metrics"]["train/epoch_mean_optimization_loss"] == 0.8
    assert events[1]["metrics"]["train/batch"] == 4
    assert events[2]["metrics"]["train/optimization_loss"] == 0.6
    assert "secret_tensor" not in json.dumps(events)
    assert "optimizer" not in json.dumps(events)
    with pytest.raises(ValueError):
        module.training_events({**saved, "losses": [float("nan"), 0.6]}, days_per_epoch=3, epochs=2)


def write_day(root, date, sse, energy, scored, committed=True):
    day = root / f"date_{date}"
    day.mkdir(parents=True)
    (day / "result.json").write_text(
        json.dumps(
            {
                "date_id": date,
                "seconds": 2.0,
                "diagnostic": {"date_id": date, "sse": sse, "denominator": energy, "rows": 1},
                "primary": {
                    "sse": sse if scored else 0.0,
                    "denominator": energy if scored else 0.0,
                    "rows": int(scored),
                },
            }
        )
    )
    if committed:
        saved = root / "checkpoints" / f"date_{date}"
        saved.mkdir(parents=True)
        (saved / "replay.json").write_text(json.dumps({"completed_date": date}))


def test_training_diagnostics_are_logged_with_true_axes_and_undefined_scores_omitted():
    saved = {
        "completed": 5,
        "epoch": 1,
        "position": 2,
        "losses": [0.4, 0.6],
        "diagnostics": [
            {"unbalanced_loss": 0.25, "responder_6_sse": 2.0, "responder_6_energy": 8.0},
            {"unbalanced_loss": None, "responder_6_sse": 1.0, "responder_6_energy": 0.0},
        ],
        "history": [
            {
                "epoch": 1,
                "mean_optimization_loss": 0.8,
                "mean_unbalanced_loss": 0.3,
                "responder_6_r2": -0.2,
            }
        ],
    }
    events = monitor().training_events(saved, days_per_epoch=3, epochs=2)
    assert events[0]["metrics"]["train/epoch_mean_unbalanced_loss"] == 0.3
    assert events[0]["metrics"]["train/epoch_responder_6_r2"] == -0.2
    assert events[1]["metrics"]["train/batch"] == 4
    assert events[1]["metrics"]["train/unbalanced_loss"] == 0.25
    assert events[1]["metrics"]["train/responder_6_r2"] == 0.75
    assert "train/responder_6_r2" not in events[2]["metrics"]
    assert "train/unbalanced_loss" not in events[2]["metrics"]
    with pytest.raises(ValueError, match="diagnostic"):
        monitor().training_events({**saved, "diagnostics": []}, days_per_epoch=3, epochs=2)
    saved["diagnostics"][0]["unbalanced_loss"] = float("nan")
    with pytest.raises(ValueError, match="diagnostic"):
        monitor().training_events(saved, days_per_epoch=3, epochs=2)


def test_eval_events_require_committed_days_and_pool_statistics_without_warmup(tmp_path):
    module = monitor()
    root = tmp_path / "offline"
    write_day(root, 10, 1, 1, False)
    write_day(root, 11, 1, 9, True)
    write_day(root, 12, 3, 1, True, committed=False)
    events = module.evaluation_events(
        tmp_path, "offline", replay_dates=(10, 11, 12), total_batches=6, window=2
    )
    assert len(events) == 2
    assert "offline/scored_cumulative_r2" not in events[0]["metrics"]
    assert "offline/rolling_r2" not in events[0]["metrics"]
    assert events[1]["metrics"]["offline/rolling_r2"] == pytest.approx(0.8)
    assert events[1]["metrics"]["offline/scored_cumulative_r2"] == pytest.approx(8 / 9)
    assert events[1]["metrics"]["offline/is_scored"] is True
    assert events[0]["step"] > 6 * 2 + 1
    assert [e["metrics"]["eval/date_id"] for e in events] == [10, 11]
    # This is observation only: parsing must not write into the training run.
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    again = module.evaluation_events(
        tmp_path, "offline", replay_dates=(10, 11, 12), total_batches=6, window=2
    )
    assert events == again
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_online_steps_follow_all_offline_dates_and_skip_incomplete_gaps(tmp_path):
    module = monitor()
    write_day(tmp_path / "online", 10, 1, 2, False)
    write_day(tmp_path / "online", 12, 1, 2, True)
    events = module.evaluation_events(
        tmp_path, "online", replay_dates=(10, 11, 12), total_batches=6, window=2
    )
    assert len(events) == 1
    assert events[0]["step"] == 17  # 2*6 + 2, plus all three offline dates.
