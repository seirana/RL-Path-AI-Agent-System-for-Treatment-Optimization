#!/usr/bin/env python3
"""Evaluate trained DQN vs baselines and save rollout summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.baselines import rollout
from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.preprocess import load_effects


def parse_kernel(arg: str | None) -> Sequence[float] | None:
    if arg is None or arg.strip() == "":
        return None
    return [float(x.strip()) for x in arg.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="RL-Path: evaluate")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_drugs", type=int, default=60)
    ap.add_argument("--top_pathways", type=int, default=40)
    ap.add_argument("--model", type=str, default="artifacts/dqn.pt")
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--n_rollouts", type=int, default=30)

    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--step_penalty", type=float, default=0.02)
    ap.add_argument("--action_cost_scale", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--disease_pathway_frac", type=float, default=0.35)
    ap.add_argument("--temporal_kernel", type=str, default="0.6,0.3,0.1")
    ap.add_argument("--interaction_scale", type=float, default=0.15)
    ap.add_argument("--interaction_history", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--replay_size", type=int, default=50_000)
    ap.add_argument("--min_replay", type=int, default=1_000)
    ap.add_argument("--target_update", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=10_000)
    ap.add_argument("--hidden_dim", type=int, default=128)
    args = ap.parse_args()

    eff_path = Path(f"data/processed/drug_pathway_effects_N{args.top_drugs}_P{args.top_pathways}.npz")
    if not eff_path.exists():
        raise FileNotFoundError(
            f"Missing effect matrix: {eff_path}. Run train.py first or place the file there."
        )
    em = load_effects(eff_path)

    env = PathwaySteeringEnv(
        effects=em.effects,
        drug_names=em.drug_names,
        pathway_names=em.pathway_names,
        steps=args.steps,
        seed=args.seed,
        alpha=args.alpha,
        step_penalty=args.step_penalty,
        action_cost_scale=args.action_cost_scale,
        noise=args.noise,
        disease_pathway_frac=args.disease_pathway_frac,
        temporal_kernel=parse_kernel(args.temporal_kernel),
        interaction_scale=args.interaction_scale,
        interaction_history=args.interaction_history,
    )

    cfg = DQNConfig(
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
    )
    agent = DQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, cfg=cfg, seed=args.seed)
    agent.load(args.model)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def dqn_rollout() -> tuple[float, list[str]]:
        obs = env.reset()
        total = 0.0
        actions: list[str] = []
        done = False
        while not done:
            a = agent.act(obs, greedy=True)
            res = env.step(a)
            total += res.reward
            actions.append(res.info["drug"])
            obs = res.obs
            done = res.done
        return float(total), actions

    dqn_runs = [dqn_rollout() for _ in range(args.n_rollouts)]
    greedy_runs = [rollout(env, policy="greedy") for _ in range(args.n_rollouts)]
    rand_runs = [rollout(env, policy="random") for _ in range(args.n_rollouts)]

    dqn_rets = [x[0] for x in dqn_runs]
    greedy_rets = [x[0] for x in greedy_runs]
    rand_rets = [x[0] for x in rand_runs]

    summary = {
        "n_rollouts": args.n_rollouts,
        "dqn": {
            "mean": float(np.mean(dqn_rets)),
            "std": float(np.std(dqn_rets)),
            "example_actions": dqn_runs[0][1] if dqn_runs else [],
        },
        "greedy": {
            "mean": float(np.mean(greedy_rets)),
            "std": float(np.std(greedy_rets)),
            "example_actions": greedy_runs[0][1] if greedy_runs else [],
        },
        "random": {
            "mean": float(np.mean(rand_rets)),
            "std": float(np.std(rand_rets)),
            "example_actions": rand_runs[0][1] if rand_runs else [],
        },
    }
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
