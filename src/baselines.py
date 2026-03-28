"""Baselines for RL-Path."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .env import PathwaySteeringEnv


def greedy_one_step(env: PathwaySteeringEnv, obs: np.ndarray) -> int:
    """Choose the action with the best immediate surrogate reward.

    This baseline is intentionally myopic: it ignores future delayed effects beyond what would be
    applied on the current transition. It still respects the updated environment structure by using
    the environment's current pending effects and pairwise interaction logic.
    """
    best_a, best_r = 0, -1e18

    s = env.state.astype(np.float32, copy=True)
    pending_now = np.zeros(env.n_pathways, dtype=np.float32)
    for item in env.pending_effects:
        if int(item["delay"]) <= 0:
            pending_now += np.asarray(item["effect"], dtype=np.float32)

    prev = float(np.mean((s[env.disease_mask] - env.target[env.disease_mask]) ** 2))

    for a in range(env.n_actions):
        parts = env.compute_action_components(a)
        # only the immediate kernel mass lands now; future delayed pieces are ignored by design
        immediate_action_piece = float(env.temporal_kernel[0]) * np.asarray(parts["scheduled_effect"], dtype=np.float32)
        applied = pending_now + immediate_action_piece
        s2 = np.clip(s - env.alpha * applied, 0.0, 1.0)
        new = float(np.mean((s2[env.disease_mask] - env.target[env.disease_mask]) ** 2))
        improvement = prev - new
        r = float(improvement - env.step_penalty - float(parts["action_cost"]))
        if r > best_r:
            best_r, best_a = r, a
    return best_a


def rollout(env: PathwaySteeringEnv, policy: str = "random") -> Tuple[float, List[str]]:
    obs = env.reset()
    total = 0.0
    actions: List[str] = []
    done = False
    while not done:
        if policy == "random":
            a = env.sample_action()
        elif policy == "greedy":
            a = greedy_one_step(env, obs)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        res = env.step(a)
        obs = res.obs
        total += res.reward
        actions.append(res.info["drug"])
        done = res.done
    return float(total), actions
