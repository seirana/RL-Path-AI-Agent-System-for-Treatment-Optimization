import numpy as np
import pytest

from src.env import PathwaySteeringEnv


def make_env(**kwargs):
    effects = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return PathwaySteeringEnv(
        effects=effects,
        drug_names=["drug_a", "drug_b"],
        pathway_names=["path_a", "path_b"],
        disease_mask=np.array([True, False]),
        noise=0.0,
        step_penalty=0.0,
        action_cost_scale=0.0,
        **kwargs,
    )


def test_delayed_effect_lands_on_next_transition():
    env = PathwaySteeringEnv(
        effects=np.array([[1.0]], dtype=np.float32),
        drug_names=["drug_a"],
        pathway_names=["path_a"],
        disease_mask=np.array([True]),
        steps=3,
        alpha=1.0,
        noise=0.0,
        step_penalty=0.0,
        action_cost_scale=0.0,
        temporal_kernel=[0.5, 0.5],
    )
    env.reset(initial_state=np.array([1.0], dtype=np.float32))

    first = env.step(0)
    assert first.info["applied_effect_sum"] == pytest.approx(0.5)
    assert first.info["pending_effect_sum"] == pytest.approx(0.5)
    assert env.state[0] == pytest.approx(0.5)

    second = env.step(0)
    assert second.info["applied_effect_sum"] == pytest.approx(1.0)
    assert env.state[0] == pytest.approx(0.0)


def test_observation_exposes_markov_context():
    env = make_env(
        temporal_kernel=[0.5, 0.3, 0.2],
        interaction_history=2,
    )
    obs = env.reset()

    expected = (
        2
        + 2
        + (2 * 2)
        + (2 * 2)
        + 1
    )
    assert env.obs_dim == expected
    assert obs.shape == (expected,)


def test_random_action_rng_does_not_change_transition_noise():
    kwargs = dict(
        effects=np.array([[0.2, 0.1]], dtype=np.float32),
        drug_names=["drug_a"],
        pathway_names=["path_a", "path_b"],
        disease_mask=np.array([True, False]),
        steps=2,
        seed=123,
        noise=0.05,
        temporal_kernel=[1.0],
    )
    env_a = PathwaySteeringEnv(**kwargs)
    env_b = PathwaySteeringEnv(**kwargs)

    start = np.array([0.8, 0.4], dtype=np.float32)
    env_a.reset(initial_state=start)
    env_b.reset(initial_state=start)

    for _ in range(20):
        env_a.sample_action()

    env_a.step(0)
    env_b.step(0)

    np.testing.assert_allclose(env_a.state, env_b.state)


def test_step_after_done_is_rejected():
    env = make_env(steps=1)
    env.reset()
    env.step(0)

    with pytest.raises(RuntimeError, match="already complete"):
        env.step(0)


def test_invalid_interaction_history_is_rejected():
    with pytest.raises(ValueError, match="interaction_history"):
        make_env(interaction_history=0)
