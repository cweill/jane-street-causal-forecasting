"""Local research commands. All fitting commands run the safety gate first."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="run metric, protocol and causal checks")
    synthetic = sub.add_parser("synthetic", help="write deterministic synthetic parquet")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--days", type=int, default=10)
    for name in ("run", "ablations", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/baseline.yaml")
        command.add_argument("--output", required=True)
        if name == "run":
            command.add_argument("--cache", help="reusable Patrick preparation cache directory")
        if name != "smoke":
            command.add_argument(
                "--data", required=True, help="train.parquet file or partition directory"
            )
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "verify":
        from src.safety import safety_gate

        print(safety_gate()["output"])
        return
    if args.command == "synthetic":
        from src.data.synthetic import synthetic_panel

        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_panel(days=args.days).write_parquet(path)
        print(path)
        return
    config = load_config(args.config)
    if args.command == "smoke":
        from src.data.loader import FrameSource
        from src.data.synthetic import synthetic_panel

        # Small dimensions and one epoch check execution only; flags are preserved.
        panel = synthetic_panel()
        if getattr(config, "method", None) == "patrick":
            import polars as pl

            model = replace(
                config.model,
                d_model=8,
                nheads=2,
                d_hidden=16,
                layers=2,
                d_cat=2,
                head_sizes=(12, 8),
            )
            features = config.features
            # The general synthetic fixture uses continuous values everywhere.
            # Patrick's three categorical columns require a finite vocabulary.
            panel = panel.with_columns(
                [
                    (pl.col("symbol_id") % 2).cast(pl.Float32).alias(c)
                    for c in features.category_columns
                ]
            )
        else:
            model = replace(
                config.model,
                hidden_sizes=(8,),
                linear_sizes=(6,),
                dropout=(0.1,),
                linear_dropout=(0.1,),
            )
            features = replace(config.features, rolling_window=3)
        config = replace(
            config,
            name=config.name + "_synthetic_smoke",
            training=replace(config.training, epochs=1, device="cpu"),
            model=model,
            features=features,
            cv=replace(
                config.cv,
                min_date=0,
                n_splits=2,
                validation_days=2,
                gap_days=0,
                train_end=None,
                warmup_end=None,
                validation_end=None,
            ),
        )
        source = FrameSource(panel)
    else:
        from src.data.loader import ParquetSource

        source = ParquetSource(args.data)
    if args.command == "ablations":
        from experiments.ablations import run_ablations

        result = run_ablations(source, config, args.output)
    else:
        from src.experiment import run_experiment

        result = run_experiment(
            source, config, args.output, cache_root=getattr(args, "cache", None)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
