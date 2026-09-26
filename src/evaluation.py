"""Evaluation helpers for paired, reproducible RL-Path comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Sequence

import numpy as np

from .baselines import greedy_one_step
from .dqn import DQNAgent
from .env import PathwaySteeringEnv


@dataclass(frozen=True)
class RolloutResult:
    total_return: float
    actions: list[str]
    initial_disease_mse: float
    final_disease_mse: float


def rollout_policy(
    env: PathwaySteeringEnv,
    *,
    policy: str,
    agent: DQNAgent | None = None,
) -> RolloutResult:
    """Run one episode for a named policy."""

    obs = env.reset()
    initial_mse = env.disease_mse()
    total = 0.0
    actions: list[str] = []
    done = False

    while not done:
        if policy == "random":
            action = env.sample_action()
        elif policy == "greedy":
            action = greedy_one_step(env, obs)
        elif policy == "dqn":
            if agent is None:
                raise ValueError(
                    "agent is required for policy='dqn'"
                )
            action = agent.act(
                obs,
                greedy=True,
            )
        else:
            raise ValueError(
                f"Unknown policy: {policy}"
            )

        result = env.step(action)
        total += result.reward
        actions.append(
            str(result.info["drug"])
        )
        obs = result.obs
        done = result.done

    return RolloutResult(
        total_return=float(total),
        actions=actions,
        initial_disease_mse=initial_mse,
        final_disease_mse=env.disease_mse(),
    )


def summary_statistics(
    values: Sequence[float],
) -> dict[str, float | int]:
    """Return mean, sample SD, SE, and a 95% normal-approximation CI."""

    array = np.asarray(
        list(values),
        dtype=float,
    )
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "values must be a non-empty 1D sequence"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "values must contain only finite values"
        )

    mean = float(array.mean())
    if array.size == 1:
        std = 0.0
        stderr = 0.0
    else:
        std = float(
            array.std(ddof=1)
        )
        stderr = std / sqrt(
            float(array.size)
        )

    half_width = 1.96 * stderr
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "stderr": float(stderr),
        "ci95_low": float(
            mean - half_width
        ),
        "ci95_high": float(
            mean + half_width
        ),
    }


def paired_difference_statistics(
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, float | int]:
    """Summarize paired first-minus-second differences."""

    a = np.asarray(
        list(first),
        dtype=float,
    )
    b = np.asarray(
        list(second),
        dtype=float,
    )
    if a.shape != b.shape:
        raise ValueError(
            "paired sequences must have the same shape"
        )
    return summary_statistics(
        (a - b).tolist()
    )


def evaluate_policies(
    *,
    make_env: Callable[[int], PathwaySteeringEnv],
    agent: DQNAgent,
    n_rollouts: int,
    seed: int,
) -> dict[str, object]:
    """Evaluate DQN, greedy, and random policies on paired simulator seeds.

    Each policy gets a fresh environment constructed with the same rollout seed.
    This keeps disease masks, initial-state draws, and transition-noise streams
    paired across policies. Random action selection uses a separate RNG stream
    inside the environment, so it does not perturb transition noise.
    """

    if n_rollouts <= 0:
        raise ValueError(
            "n_rollouts must be greater than 0"
        )

    policy_names = (
        "dqn",
        "greedy",
        "random",
    )
    results: dict[
        str,
        list[RolloutResult],
    ] = {
        name: []
        for name in policy_names
    }

    rollout_seeds = [
        int(seed + index)
        for index in range(n_rollouts)
    ]

    for rollout_seed in rollout_seeds:
        for policy in policy_names:
            env = make_env(rollout_seed)
            results[policy].append(
                rollout_policy(
                    env,
                    policy=policy,
                    agent=(
                        agent
                        if policy == "dqn"
                        else None
                    ),
                )
            )

    summary: dict[str, object] = {
        "n_rollouts": n_rollouts,
        "rollout_seeds": rollout_seeds,
        "policies": {},
        "paired_return_differences": {},
        "ci_method": (
            "95% normal approximation using "
            "sample standard error"
        ),
    }

    returns_by_policy: dict[
        str,
        list[float],
    ] = {}

    for policy in policy_names:
        policy_results = results[policy]
        returns = [
            item.total_return
            for item in policy_results
        ]
        returns_by_policy[policy] = returns

        final_mse = [
            item.final_disease_mse
            for item in policy_results
        ]

        summary["policies"][policy] = {
            "return": summary_statistics(
                returns
            ),
            "final_disease_mse": (
                summary_statistics(
                    final_mse
                )
            ),
            "example_actions": (
                policy_results[0].actions
                if policy_results
                else []
            ),
        }

    summary[
        "paired_return_differences"
    ] = {
        "dqn_minus_greedy": (
            paired_difference_statistics(
                returns_by_policy["dqn"],
                returns_by_policy["greedy"],
            )
        ),
        "dqn_minus_random": (
            paired_difference_statistics(
                returns_by_policy["dqn"],
                returns_by_policy["random"],
            )
        ),
    }

    return summary
