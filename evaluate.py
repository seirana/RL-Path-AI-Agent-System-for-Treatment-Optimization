#!/usr/bin/env python3
"""Evaluate a trained DQN against paired simulator baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.evaluation import evaluate_policies
from src.preprocess import EffectMatrix, load_effects


def parse_kernel(arg: str | None) -> Sequence[float] | None:
    if arg is None or arg.strip() == "":
        return None
    values = [
        float(value.strip())
        for value in arg.split(",")
        if value.strip()
    ]
    if not values:
        return None
    return values


def build_env(
    args: argparse.Namespace,
    effect_matrix: EffectMatrix,
    *,
    seed: int,
) -> PathwaySteeringEnv:
    return PathwaySteeringEnv(
        effects=effect_matrix.effects,
        drug_names=effect_matrix.drug_names,
        pathway_names=effect_matrix.pathway_names,
        steps=args.steps,
        seed=seed,
        alpha=args.alpha,
        step_penalty=args.step_penalty,
        action_cost_scale=args.action_cost_scale,
        noise=args.noise,
        disease_pathway_frac=args.disease_pathway_frac,
        temporal_kernel=parse_kernel(args.temporal_kernel),
        interaction_scale=args.interaction_scale,
        interaction_history=args.interaction_history,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RL-Path policies on paired simulator seeds"
        )
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--top_drugs", type=int, default=60)
    parser.add_argument("--top_pathways", type=int, default=40)
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/dqn.pt"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=30,
    )

    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument(
        "--step_penalty",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--action_cost_scale",
        type=float,
        default=0.05,
    )
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument(
        "--disease_pathway_frac",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--temporal_kernel",
        type=str,
        default="0.6,0.3,0.1",
    )
    parser.add_argument(
        "--interaction_scale",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--interaction_history",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--replay_size",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--min_replay",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--eps_start",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--eps_end",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--eps_decay_steps",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=5.0,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_rollouts <= 0:
        raise ValueError(
            "n_rollouts must be greater than 0"
        )

    effect_path = args.processed_dir / (
        f"drug_pathway_effects_N{args.top_drugs}_"
        f"P{args.top_pathways}.npz"
    )
    if not effect_path.exists():
        raise FileNotFoundError(
            f"Missing effect matrix: {effect_path}. "
            "Run train.py first or create the processed matrix."
        )

    effect_matrix = load_effects(effect_path)
    reference_env = build_env(
        args,
        effect_matrix,
        seed=args.seed,
    )

    config = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay=args.min_replay,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        hidden_dim=args.hidden_dim,
        grad_clip_norm=args.grad_clip_norm,
    )
    agent = DQNAgent(
        obs_dim=reference_env.obs_dim,
        n_actions=reference_env.n_actions,
        cfg=config,
        seed=args.seed,
    )
    agent.load(args.model)

    summary = evaluate_policies(
        make_env=lambda rollout_seed: build_env(
            args,
            effect_matrix,
            seed=rollout_seed,
        ),
        agent=agent,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        args.outdir
        / "eval_summary.json"
    )
    output_path.write_text(
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
