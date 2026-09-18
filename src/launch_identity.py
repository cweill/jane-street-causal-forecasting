"""Compare persisted launch manifests using the same JSON representation as storage."""

import json
from pathlib import Path


def validate_launch_identity(path, identity):
    canonical = json.loads(json.dumps(identity, allow_nan=False))
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != canonical:
        raise ValueError("launch identity changed; restore the recorded source/data to resume")
    return canonical
