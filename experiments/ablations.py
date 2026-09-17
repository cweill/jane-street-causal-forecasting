"""Matched one-factor ablations. Changes never compound across comparisons."""

from dataclasses import replace

from src.artifacts import write_json
from src.experiment import _run_verified
from src.safety import safety_gate

SWITCHES = {
    "market_average": ("features", "market_average"),
    "rolling": ("features", "rolling"),
    "auxiliary_targets": ("model", "auxiliary_targets"),
    "online_updates": ("online", "enabled"),
    "seed_ensembling": ("ensemble", "seed_ensembling"),
}


def ablation_configs(reference):
    """Reference plus five independent on/off reversals, not a cumulative ladder."""
    yield "reference", reference
    for name, (section, field) in SWITCHES.items():
        original = getattr(reference, section)
        value = not getattr(original, field)
        label = f"{name}_{'on' if value else 'off'}"
        yield (
            label,
            replace(reference, name=label, **{section: replace(original, **{field: value})}),
        )


def run_ablations(source, reference, output):
    from pathlib import Path

    gate = safety_gate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, config in ablation_configs(reference):
        result = _run_verified(source, config, output / label, gate)
        rows.append(
            {
                "ablation": label,
                "pooled_score": result["pooled_score"],
                "delta_from_reference": result["pooled_score"] - rows[0]["pooled_score"]
                if rows
                else 0.0,
            }
        )
        write_json(output / "comparison.json", rows)
    return rows
