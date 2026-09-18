import importlib
import importlib.util
import json

import pytest

from src.config import load_config


def test_launch_identity_survives_json_roundtrip_but_rejects_real_changes(tmp_path):
    assert importlib.util.find_spec("src.launch_identity"), "canonical launch comparison missing"
    module = importlib.import_module("src.launch_identity")
    identity = {
        "config": load_config("configs/patrick_ol.yaml").to_dict(),
        "dataset_sha256": "a" * 64,
        "code_sha256": "b" * 64,
    }
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(identity))
    # Dataclass config contains tuples; JSON replaces them with lists.
    assert json.loads(path.read_text()) != identity
    module.validate_launch_identity(path, identity)
    for altered in [
        {**identity, "dataset_sha256": "c" * 64},
        {**identity, "code_sha256": "d" * 64},
        {**identity, "config": {**identity["config"], "name": "different-experiment"}},
    ]:
        with pytest.raises(ValueError, match="identity"):
            module.validate_launch_identity(path, altered)
