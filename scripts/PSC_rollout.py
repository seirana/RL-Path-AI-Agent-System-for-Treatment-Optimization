#!/usr/bin/env python3
"""Run a PSC-focused greedy DQN rollout from a custom initial pathway state."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.preprocess import load_effects

TOP_DRUGS = 60
TOP_PATHWAYS = 40
STEPS = 10
PSC_KEYWORDS = {
    "immune": [
        "MHC",
        "antigen",
        "interferon",
        "IFN",
        "TNF",
        "NF-kB",
        "NFκB",
        "T cell",
        "T-cell",
        "macrophage",
    ],
    "fibrosis": [
        "TGF",
        "ECM",
        "extracellular matrix",
        "collagen",
        "wound",
        "cholangiocyte",
        "proliferation",
    ],
}


def find_matches(pathway_names, keywords):
    hits = []
    for i, name in enumerate(pathway_names):
        for kw in keywords:
            if re.search(re.escape(kw), name, flags=re.IGNORECASE):
                hits.append((i, name, kw))
                break
    return hits


def main() -> None:
    npz_path = Path(f"./data/processed/drug_pathway_effects_N{TOP_DRUGS}_P{TOP_PATHWAYS}.npz")
    em = load_effects(npz_path)
    env = PathwaySteeringEnv(
        effects=em.effects,
        drug_names=em.drug_names,
        pathway_names=em.pathway_names,
        steps=STEPS,
        seed=42,
        temporal_kernel=[0.6, 0.3, 0.1],
        interaction_scale=0.15,
    )
    agent = DQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, cfg=DQNConfig(), seed=42)
    agent.load("artifacts/dqn.pt")

    immune_hits = find_matches(env.pathway_names, PSC_KEYWORDS["immune"])
    fibrosis_hits = find_matches(env.pathway_names, PSC_KEYWORDS["fibrosis"])

    print("\n=== Matched IMMUNE-related pathways in training set ===")
    for i, name, kw in immune_hits:
        print(f"[{i:02d}] {name} (matched: {kw})")

    print("\n=== Matched FIBROSIS-related pathways in training set ===")
    for i, name, kw in fibrosis_hits:
        print(f"[{i:02d}] {name} (matched: {kw})")

    if not immune_hits and not fibrosis_hits:
        print("\nNo matches found in the selected TOP_PATHWAYS list.")
        print("=> Increase --top_pathways or build a PSC-specific pathway panel and retrain.")
        return

    env.reset()
    start = np.full(env.n_pathways, 0.35, dtype=np.float32)
    for i, _, _ in immune_hits:
        start[i] = 0.90
    for i, _, _ in fibrosis_hits:
        start[i] = 0.85
    env.state = start.copy()
    env.pending_effects = []
    env.action_history = []
    env.last_action = None

    obs = env._get_obs()
    seq = []
    done = False
    while not done:
        a = agent.act(obs, greedy=True)
        step = env.step(a)
        seq.append(step.info["drug"])
        obs = step.obs
        done = step.done

    print("\n=== Suggested drug order (DQN greedy rollout) ===")
    for t, d in enumerate(seq, 1):
        print(f"{t:02d}. {d}")


if __name__ == "__main__":
    main()

