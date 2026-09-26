import numpy as np
import pytest

from src.dqn import DQNAgent, DQNConfig, ReplayBuffer


def test_dqn_config_validation():
    with pytest.raises(ValueError, match="gamma"):
        DQNConfig(gamma=1.5)


def test_replay_buffer_sampling_is_seeded():
    first = ReplayBuffer(capacity=10, seed=9)
    second = ReplayBuffer(capacity=10, seed=9)

    for index in range(8):
        state = np.array([index], dtype=np.float32)
        next_state = np.array([index + 1], dtype=np.float32)
        first.push(state, index % 2, float(index), next_state, False)
        second.push(state, index % 2, float(index), next_state, False)

    batch_a = first.sample(4)
    batch_b = second.sample(4)

    for left, right in zip(batch_a, batch_b, strict=True):
        np.testing.assert_array_equal(left, right)


def test_agent_checkpoint_records_shape_metadata(tmp_path):
    cfg = DQNConfig(
        batch_size=2,
        replay_size=10,
        min_replay=2,
        hidden_dim=8,
    )
    agent = DQNAgent(
        obs_dim=3,
        n_actions=2,
        cfg=cfg,
        seed=7,
    )
    path = tmp_path / "agent.pt"
    agent.save(path)

    compatible = DQNAgent(
        obs_dim=3,
        n_actions=2,
        cfg=cfg,
        seed=7,
    )
    compatible.load(path)

    incompatible = DQNAgent(
        obs_dim=4,
        n_actions=2,
        cfg=cfg,
        seed=7,
    )
    with pytest.raises(ValueError, match="observation dimension"):
        incompatible.load(path)
