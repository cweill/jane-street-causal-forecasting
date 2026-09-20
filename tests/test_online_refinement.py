import importlib
import importlib.util
import json
from dataclasses import asdict

import pytest
from test_online_sweep import fixture

from src.online_sweep import Trial, run_trial, summarize_trials


def test_refinement_reuses_completed_trials_with_explicit_provenance_and_block_scores(tmp_path):
    assert importlib.util.find_spec("src.online_refinement"), "refinement support missing"
    api = importlib.import_module("src.online_refinement")
    source, base, fold = fixture(tmp_path)
    frozen = Trial("frozen", None, False)
    old = Trial("lr_1e-4_persistent", 1e-4, False)
    new = Trial("lr_5e-5_persistent", 5e-5, False)
    roots = {t.name: tmp_path / t.name for t in (frozen, old, new)}
    provenance = {
        t.name: {"dataset": "same", "code": "new" if t == new else "old"}
        for t in (frozen, old, new)
    }
    for t in (frozen, old, new):
        run_trial(source, base, roots[t.name], t, fold, device="cpu", provenance=provenance[t.name])
    immutable = {
        str(p): p.read_bytes()
        for t in (frozen, old)
        for p in roots[t.name].rglob("*")
        if p.is_file()
    }
    report = summarize_trials(
        tmp_path / "report",
        [frozen, old, new],
        fold,
        window=2,
        trial_directories=roots,
        expected_provenance=provenance,
    )
    assert len(report["trials"]) == 3
    blocks = api.block_comparison(roots, [frozen, old, new], fold, block_days=2)
    assert blocks.height == 6
    assert blocks.filter(blocks["trial"] == "frozen")["delta_r2"].to_list() == [0.0, 0.0]
    assert sorted(blocks["days"].unique().to_list()) == [1, 2]
    assert immutable == {
        str(p): p.read_bytes()
        for t in (frozen, old)
        for p in roots[t.name].rglob("*")
        if p.is_file()
    }
    bad = {**provenance, new.name: {"dataset": "different", "code": "new"}}
    with pytest.raises(ValueError, match="paired"):
        summarize_trials(
            tmp_path / "bad",
            [frozen, old, new],
            fold,
            window=2,
            trial_directories=roots,
            expected_provenance=bad,
        )
    assert [t.learning_rate for t in api.new_trials()] == [1e-5, 3e-5, 5e-5, 2e-4]
    assert all(not t.reset_daily for t in api.new_trials())
    assert [asdict(t) for t in api.reused_trials()] == [asdict(frozen), asdict(old)]
    p = roots[new.name] / "online/date_4/result.json"
    d = json.loads(p.read_text())
    d["primary"]["rows"] += 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="coverage"):
        api.block_comparison(roots, [frozen, old, new], fold, block_days=2)
