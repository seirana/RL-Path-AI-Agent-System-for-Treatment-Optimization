import numpy as np
import pytest

from src.env import PathwaySteeringEnv
from src.evaluation import (
    evaluate_policies,
    paired_difference_statistics,
    summary_statistics,
)


class ZeroAgent:
    def act(self, obs, greedy=False):
        return 0


def make_env(seed):
    return PathwaySteeringEnv(
        effects=np.array(
            [
                [0.7, 0.0],
                [0.0, 0.2],
            ],
            dtype=np.float32,
        ),
        drug_names=["strong", "weak"],
        pathway_names=["disease", "other"],
        disease_mask=np.array([True, False]),
        steps=2,
        seed=seed,
        noise=0.01,
        temporal_kernel=[1.0],
    )


def test_summary_statistics_reports_ci():
    stats = summary_statistics([1.0, 2.0, 3.0])

    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["ci95_low"] < stats["mean"]
    assert stats["ci95_high"] > stats["mean"]


def test_paired_difference_uses_matching_positions():
    stats = paired_difference_statistics(
        [3.0, 4.0],
        [1.0, 1.0],
    )

    assert stats["mean"] == pytest.approx(2.5)


def test_policy_evaluation_uses_requested_rollout_count():
    summary = evaluate_policies(
        make_env=make_env,
        agent=ZeroAgent(),
        n_rollouts=4,
        seed=100,
    )

    assert summary["n_rollouts"] == 4
    assert summary["rollout_seeds"] == [100, 101, 102, 103]
    assert summary["policies"]["dqn"]["return"]["n"] == 4
    assert "dqn_minus_greedy" in summary["paired_return_differences"]
