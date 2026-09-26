#!/usr/bin/env python3
"""Train DQN on the RL-Path pathway-steering simulator."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.evaluation import evaluate_policies
from src.preprocess import (
    EffectMatrix,
    build_effect_matrix,
    load_dgidb_interactions,
    load_effects,
    load_reactome_ensembl2reactome,
    save_effects,
)


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


def ensure_effects(
    top_drugs: int,
    top_pathways: int,
    raw_dir: Path,
    processed_dir: Path,
) -> Path:
    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    outpath = processed_dir / (
        f"drug_pathway_effects_N{top_drugs}_"
        f"P{top_pathways}.npz"
    )
    if outpath.exists():
        return outpath

    dgidb = raw_dir / "dgidb_interactions.tsv"
    reactome = raw_dir / "Ensembl2Reactome.txt"
    if not dgidb.exists() or not reactome.exists():
        raise FileNotFoundError(
            f"Missing raw data under {raw_dir}. Expected "
            "dgidb_interactions.tsv and Ensembl2Reactome.txt."
        )

    dgidb_df = load_dgidb_interactions(dgidb)
    reactome_df = load_reactome_ensembl2reactome(
        reactome
    )
    effect_matrix = build_effect_matrix(
        dgidb_df,
        reactome_df,
        top_drugs=top_drugs,
        top_pathways=top_pathways,
        symbol_cache_path=(
            processed_dir
            / "symbol_to_ensembl.tsv"
        ),
    )
    save_effects(
        effect_matrix,
        outpath,
    )
    return outpath


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
        disease_pathway_frac=(
            args.disease_pathway_frac
        ),
        temporal_kernel=parse_kernel(
            args.temporal_kernel
        ),
        interaction_scale=(
            args.interaction_scale
        ),
        interaction_history=(
            args.interaction_history
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RL-Path DQN training on a simulated "
            "pathway-steering environment"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--eval_rollouts",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--top_drugs",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--top_pathways",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts"),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
    )
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
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--disease_pathway_frac",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--temporal_kernel",
        type=str,
        default="0.6,0.3,0.1",
        help=(
            "Comma-separated delayed-effect kernel. "
            "Example: 0.6,0.3,0.1"
        ),
    )
    parser.add_argument(
        "--interaction_scale",
        type=float,
        default=0.15,
        help=(
            "Scale for heuristic pairwise overlap effects. "
            "This is a simulator parameter, not a clinical "
            "drug-interaction estimate."
        ),
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

    if args.episodes <= 0:
        raise ValueError(
            "episodes must be greater than 0"
        )
    if args.eval_rollouts <= 0:
        raise ValueError(
            "eval_rollouts must be greater than 0"
        )

    effect_path = ensure_effects(
        args.top_drugs,
        args.top_pathways,
        args.raw_dir,
        args.processed_dir,
    )
    effect_matrix = load_effects(
        effect_path
    )
    env = build_env(
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
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        cfg=config,
        seed=args.seed,
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    returns: list[float] = []
    losses: list[float] = []

    minimum_buffer = max(
        config.min_replay,
        config.batch_size,
    )
    warmup_episodes = max(
        1,
        math.ceil(
            minimum_buffer
            / env.max_steps
        ),
    )

    for _ in range(warmup_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.sample_action()
            result = env.step(action)
            agent.push(
                obs,
                action,
                result.reward,
                result.obs,
                result.done,
            )
            obs = result.obs
            done = result.done

    for episode in range(
        1,
        args.episodes + 1,
    ):
        obs = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action = agent.act(obs)
            result = env.step(action)
            agent.push(
                obs,
                action,
                result.reward,
                result.obs,
                result.done,
            )
            update = agent.update()
            loss = update.get(
                "loss",
                float("nan"),
            )
            if not np.isnan(loss):
                losses.append(
                    float(loss)
                )

            obs = result.obs
            episode_return += result.reward
            done = result.done

        returns.append(
            float(episode_return)
        )

        if episode % 50 == 0:
            print(
                f"ep={episode:4d} "
                f"return={np.mean(returns[-20:]): .4f} "
                f"eps={agent.epsilon():.3f}"
            )

    checkpoint_path = (
        args.outdir / "dqn.pt"
    )
    agent.save(checkpoint_path)

    evaluation = evaluate_policies(
        make_env=lambda rollout_seed: build_env(
            args,
            effect_matrix,
            seed=rollout_seed,
        ),
        agent=agent,
        n_rollouts=args.eval_rollouts,
        seed=args.eval_seed,
    )

    metrics = {
        "episodes": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "eval_rollouts": args.eval_rollouts,
        "top_drugs": args.top_drugs,
        "top_pathways": args.top_pathways,
        "obs_dim": env.obs_dim,
        "n_actions": env.n_actions,
        "return_mean_last20": float(
            np.mean(returns[-20:])
        ),
        "loss_mean_last100": (
            float(
                np.mean(losses[-100:])
            )
            if losses
            else None
        ),
        "environment": {
            "temporal_kernel": [
                float(value)
                for value in env.temporal_kernel
            ],
            "interaction_scale": (
                args.interaction_scale
            ),
            "interaction_history": (
                env.interaction_history
            ),
            "disease_pathway_frac": (
                args.disease_pathway_frac
            ),
            "noise": args.noise,
            "alpha": args.alpha,
            "step_penalty": (
                args.step_penalty
            ),
            "action_cost_scale": (
                args.action_cost_scale
            ),
        },
        "evaluation": evaluation,
    }

    (
        args.outdir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        args.outdir / "returns.json"
    ).write_text(
        json.dumps(
            returns,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        args.outdir / "losses.json"
    ).write_text(
        json.dumps(
            losses,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        args.outdir
        / "eval_summary.json"
    ).write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )

    run_metadata = {
        "command": "train",
        "arguments": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "checkpoint": str(
            checkpoint_path
        ),
        "effect_matrix": str(
            effect_path
        ),
    }
    (
        args.outdir
        / "run_metadata.json"
    ).write_text(
        json.dumps(
            run_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(
        "DQN Training Return"
    )
    plt.savefig(
        args.outdir
        / "learning_curve.png",
        bbox_inches="tight",
    )
    plt.close()

    print(
        "Done. Artifacts in:",
        args.outdir.resolve(),
    )


if __name__ == "__main__":
    main()
