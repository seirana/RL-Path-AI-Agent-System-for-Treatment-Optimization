#!/usr/bin/env python3
"""Run a PSC-focused simulated DQN rollout.

This script is a research demonstration. It does not produce a treatment
recommendation and the simulated drug/pathway effects are not clinically
validated intervention effects.
"""

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
    for index, name in enumerate(pathway_names):
        for keyword in keywords:
            if re.search(
                re.escape(keyword),
                name,
                flags=re.IGNORECASE,
            ):
                hits.append(
                    (
                        index,
                        name,
                        keyword,
                    )
                )
                break
    return hits


def main() -> None:
    npz_path = Path(
        "./data/processed/"
        f"drug_pathway_effects_N{TOP_DRUGS}_"
        f"P{TOP_PATHWAYS}.npz"
    )
    effect_matrix = load_effects(npz_path)

    immune_hits = find_matches(
        effect_matrix.pathway_names,
        PSC_KEYWORDS["immune"],
    )
    fibrosis_hits = find_matches(
        effect_matrix.pathway_names,
        PSC_KEYWORDS["fibrosis"],
    )

    print(
        "\n=== Matched immune-related pathways "
        "in the simulator panel ==="
    )
    for index, name, keyword in immune_hits:
        print(
            f"[{index:02d}] {name} "
            f"(matched: {keyword})"
        )

    print(
        "\n=== Matched fibrosis-related pathways "
        "in the simulator panel ==="
    )
    for index, name, keyword in fibrosis_hits:
        print(
            f"[{index:02d}] {name} "
            f"(matched: {keyword})"
        )

    matched_indices = sorted(
        {
            index
            for index, _, _ in (
                immune_hits
                + fibrosis_hits
            )
        }
    )
    if not matched_indices:
        print(
            "\nNo keyword matches found in the selected "
            "pathway panel."
        )
        print(
            "Increase TOP_PATHWAYS or construct a "
            "separately validated PSC-specific panel."
        )
        return

    disease_mask = np.zeros(
        len(effect_matrix.pathway_names),
        dtype=bool,
    )
    disease_mask[matched_indices] = True

    env = PathwaySteeringEnv(
        effects=effect_matrix.effects,
        drug_names=effect_matrix.drug_names,
        pathway_names=effect_matrix.pathway_names,
        steps=STEPS,
        seed=42,
        temporal_kernel=[
            0.6,
            0.3,
            0.1,
        ],
        interaction_scale=0.15,
        disease_mask=disease_mask,
    )

    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        cfg=DQNConfig(),
        seed=42,
    )
    agent.load("artifacts/dqn.pt")

    start = np.full(
        env.n_pathways,
        0.35,
        dtype=np.float32,
    )
    for index, _, _ in immune_hits:
        start[index] = 0.90
    for index, _, _ in fibrosis_hits:
        start[index] = 0.85

    obs = env.reset(
        initial_state=start
    )
    sequence = []
    done = False

    while not done:
        action = agent.act(
            obs,
            greedy=True,
        )
        step = env.step(action)
        sequence.append(
            step.info["drug"]
        )
        obs = step.obs
        done = step.done

    print(
        "\n=== Simulated DQN action sequence ==="
    )
    print(
        "Research simulation only; this is not a "
        "treatment recommendation."
    )
    for step_number, drug in enumerate(
        sequence,
        1,
    ):
        print(
            f"{step_number:02d}. {drug}"
        )


if __name__ == "__main__":
    main()
