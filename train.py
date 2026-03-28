#!/usr/bin/env python3
"""Train DQN on the pathway-steering environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.baselines import rollout
from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.preprocess import (
    build_effect_matrix,
    load_dgidb_interactions,
    load_effects,
    load_reactome_ensembl2reactome,
    save_effects,
)


def parse_kernel(arg: str | None) -> Sequence[float] | None:
    if arg is None or arg.strip() == "":
        return None
    return [float(x.strip()) for x in arg.split(",") if x.strip()]


def ensure_effects(top_drugs: int, top_pathways: int, seed: int, raw_dir: Path) -> Path:
    proc = Path("data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    outpath = proc / f"drug_pathway_effects_N{top_drugs}_P{top_pathways}.npz"
    if outpath.exists():
        return outpath

    dgidb = raw_dir / "dgidb_interactions.tsv"
    react = raw_dir / "Ensembl2Reactome.txt"
    if not dgidb.exists() or not react.exists():
        raise FileNotFoundError(
            f"Missing raw data under {raw_dir}. Expected dgidb_interactions.tsv and Ensembl2Reactome.txt."
        )

    dgidb_df = load_dgidb_interactions(dgidb)
    react_df = load_reactome_ensembl2reactome(react)
    em = build_effect_matrix(
        dgidb_df,
        react_df,
        top_drugs=top_drugs,
        top_pathways=top_pathways,
        seed=seed,
    )
    save_effects(em, proc)
    # keep backward-compatible filename with N/P suffix as well
    np.savez_compressed(
        outpath,
        effects=em.effects,
        drug_names=np.array(em.drug_names, dtype=object),
        pathway_ids=np.array(em.pathway_ids, dtype=object),
        pathway_names=np.array(em.pathway_names, dtype=object),
    )
    return outpath


def build_env(args: argparse.Namespace, effects_path: Path) -> PathwaySteeringEnv:
    em = load_effects(effects_path)
    return PathwaySteeringEnv(
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


def dqn_rollout(env: PathwaySteeringEnv, agent: DQNAgent) -> tuple[float, list[str]]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="RL-Path: DQN training")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_drugs", type=int, default=60)
    ap.add_argument("--top_pathways", type=int, default=40)
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--outdir", type=str, default="artifacts")

    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--step_penalty", type=float, default=0.02)
    ap.add_argument("--action_cost_scale", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--disease_pathway_frac", type=float, default=0.35)
    ap.add_argument(
        "--temporal_kernel",
        type=str,
        default="0.6,0.3,0.1",
        help="Comma-separated delayed-effect kernel. Example: 0.6,0.3,0.1",
    )
    ap.add_argument(
        "--interaction_scale",
        type=float,
        default=0.15,
        help="Scale for heuristic pairwise synergy/antagonism. Use negative values for antagonism.",
    )
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

    raw_dir = Path(args.raw_dir)
    eff_path = ensure_effects(args.top_drugs, args.top_pathways, args.seed, raw_dir)
    env = build_env(args, eff_path)

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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    returns: list[float] = []
    losses: list[float] = []

    for _ in range(max(50, args.min_replay // max(args.steps, 1))):
        obs = env.reset()
        done = False
        while not done:
            a = env.sample_action()
            res = env.step(a)
            agent.push(obs, a, res.reward, res.obs, res.done)
            obs = res.obs
            done = res.done

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(obs)
            res = env.step(a)
            agent.push(obs, a, res.reward, res.obs, res.done)
            upd = agent.update()
            if not np.isnan(upd.get("loss", np.nan)):
                losses.append(float(upd["loss"]))
            obs = res.obs
            ep_ret += res.reward
            done = res.done

        returns.append(float(ep_ret))
        if ep % 50 == 0:
            print(f"ep={ep:4d} return={np.mean(returns[-20:]): .4f} eps={agent.epsilon():.3f}")

    agent.save(str(outdir / "dqn.pt"))

    greedy_ret, greedy_actions = rollout(env, policy="greedy")
    rand_ret, rand_actions = rollout(env, policy="random")
    dqn_ret, dqn_actions = dqn_rollout(env, agent)

    metrics = {
        "episodes": args.episodes,
        "steps": args.steps,
        "top_drugs": args.top_drugs,
        "top_pathways": args.top_pathways,
        "seed": args.seed,
        "obs_dim": env.obs_dim,
        "n_actions": env.n_actions,
        "return_mean_last20": float(np.mean(returns[-20:])),
        "loss_mean_last100": float(np.mean(losses[-100:])) if losses else None,
        "greedy_return": float(greedy_ret),
        "random_return": float(rand_ret),
        "dqn_greedy_return": float(dqn_ret),
        "environment": {
            "temporal_kernel": list(map(float, env.temporal_kernel.tolist())),
            "interaction_history": int(env.interaction_history),
            "include_last_action": bool(env.include_last_action),
            "include_pending_summary": bool(env.include_pending_summary),
        },
        "example_actions": {
            "greedy": greedy_actions,
            "random": rand_actions,
            "dqn": dqn_actions,
        },
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "returns.json").write_text(json.dumps(returns, indent=2), encoding="utf-8")
    (outdir / "losses.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")

    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training Return")
    plt.savefig(outdir / "learning_curve.png", bbox_inches="tight")
    plt.close()

    print("Done. Artifacts in:", outdir.resolve())


if __name__ == "__main__":
    main()
