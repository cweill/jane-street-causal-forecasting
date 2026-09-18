"""Integration regression: a deployed job completes after its submitting process exits.

Run after deploying scripts/modal_patrick_reproduction.py. Uses a ten-second CPU
probe, no GPU or competition data. Evidence is written under artifacts/.
"""

import importlib
import json
import subprocess
import sys
import uuid
from pathlib import Path


def main():
    module = importlib.import_module("scripts.modal_patrick_reproduction")
    assert hasattr(module, "submit_deployed"), "persistent deployed-job submission is missing"
    import modal

    token = uuid.uuid4().hex
    output = Path("artifacts/modal-submission") / token
    output.mkdir(parents=True)
    record = output / "submission.json"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.modal_patrick_reproduction import submit_deployed; "
                "import sys; submit_deployed('submission_probe', sys.argv[1], record_path=sys.argv[2])"
            ),
            token,
            str(record),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    submitted = json.loads(record.read_text())
    call = modal.FunctionCall.from_id(submitted["function_call_id"])
    # The submitting process has exited. The delayed job must still be pending.
    try:
        call.get(timeout=0)
    except TimeoutError:
        pass
    else:
        raise AssertionError("probe was not pending after the submitter exited")
    result = call.get(timeout=60)
    assert result == {"token": token, "status": "completed_after_delay"}
    evidence = {
        **submitted,
        "submitter_exit_code": child.returncode,
        "pending_after_submitter_exit": True,
        "result": result,
        "passed": True,
    }
    (output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"artifact": str(output / "result.json"), **evidence}, indent=2))


if __name__ == "__main__":
    main()
