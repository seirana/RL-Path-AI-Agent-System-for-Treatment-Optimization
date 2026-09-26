#!/usr/bin/env python3
"""Run independent RL-Path training seeds and summarize evaluation returns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.evaluation import summary_statistics


def parse_seeds(value: str) -> list[int]:
    seeds = [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if not seeds:
        raise argparse.ArgumentTypeError(
            "At least one seed is required."
        )
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "Seeds must be unique."
        )
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run train.py for multiple independent "
            "seeds and aggregate DQN evaluation returns."
        )
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("1,2,3"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("artifacts/seed_sweep"),
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help=(
            "Additional train.py arguments after '--'. "
            "Do not pass --seed or --outdir."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    forwarded = list(args.train_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    forbidden = {
        "--seed",
        "--outdir",
    }
    if any(
        item.split("=", 1)[0]
        in forbidden
        for item in forwarded
    ):
        raise ValueError(
            "Do not pass --seed or --outdir in train_args; "
            "the sweep controls them."
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = []
    dqn_means = []

    for seed in args.seeds:
        run_dir = (
            args.output_root
            / f"seed_{seed}"
        )
        command = [
            sys.executable,
            "train.py",
            "--seed",
            str(seed),
            "--outdir",
            str(run_dir),
            *forwarded,
        ]
        print(
            "[run]",
            " ".join(command),
        )
        subprocess.run(
            command,
            check=True,
        )

        metrics_path = (
            run_dir / "metrics.json"
        )
        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )
        dqn_mean = float(
            metrics["evaluation"][
                "policies"
            ]["dqn"]["return"]["mean"]
        )
        dqn_means.append(dqn_mean)
        runs.append(
            {
                "seed": seed,
                "directory": str(
                    run_dir
                ),
                "dqn_evaluation_return_mean": (
                    dqn_mean
                ),
            }
        )

    summary = {
        "seeds": args.seeds,
        "runs": runs,
        "dqn_evaluation_return_across_training_seeds": (
            summary_statistics(
                dqn_means
            )
        ),
    }
    output = (
        args.output_root
        / "seed_sweep_summary.json"
    )
    output.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
