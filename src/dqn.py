"""Minimal DQN (Deep Q-Network) for discrete-action environments."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.98
    lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    min_replay: int = 1_000
    target_update: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    hidden_dim: int = 128
    grad_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.lr <= 0:
            raise ValueError("lr must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.replay_size <= 0:
            raise ValueError("replay_size must be greater than 0")
        if self.min_replay < 0:
            raise ValueError("min_replay must be non-negative")
        if self.target_update <= 0:
            raise ValueError("target_update must be greater than 0")
        if not 0.0 <= self.eps_end <= 1.0:
            raise ValueError("eps_end must be in [0, 1]")
        if not 0.0 <= self.eps_start <= 1.0:
            raise ValueError("eps_start must be in [0, 1]")
        if self.eps_end > self.eps_start:
            raise ValueError("eps_end must not exceed eps_start")
        if self.eps_decay_steps <= 0:
            raise ValueError("eps_decay_steps must be greater than 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be greater than 0")


Transition = Tuple[
    np.ndarray,
    int,
    float,
    np.ndarray,
    bool,
]


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self.buf: Deque[Transition] = deque(
            maxlen=capacity
        )
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append(
            (
                s.astype(np.float32),
                int(a),
                float(r),
                s2.astype(np.float32),
                bool(done),
            )
        )

    def __len__(self) -> int:
        return len(self.buf)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )
        if batch_size > len(self.buf):
            raise ValueError(
                f"batch_size={batch_size} exceeds "
                f"replay size={len(self.buf)}"
            )

        idx = self.rng.choice(
            len(self.buf),
            size=batch_size,
            replace=False,
        )
        batch = [self.buf[int(i)] for i in idx]
        s, a, r, s2, d = zip(*batch, strict=True)

        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DQNConfig,
        seed: int = 0,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError("obs_dim must be greater than 0")
        if n_actions <= 0:
            raise ValueError("n_actions must be greater than 0")

        self.cfg = cfg
        self.seed = int(seed)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(self.seed)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.q = QNet(
            self.obs_dim,
            self.n_actions,
            hidden=cfg.hidden_dim,
        ).to(self.device)
        self.q_tgt = QNet(
            self.obs_dim,
            self.n_actions,
            hidden=cfg.hidden_dim,
        ).to(self.device)
        self.q_tgt.load_state_dict(
            self.q.state_dict()
        )

        self.optim = optim.Adam(
            self.q.parameters(),
            lr=cfg.lr,
        )
        self.rb = ReplayBuffer(
            cfg.replay_size,
            seed=self.seed + 1,
        )
        self.step = 0

    def epsilon(self) -> float:
        t = min(
            self.step,
            self.cfg.eps_decay_steps,
        )
        frac = t / float(
            self.cfg.eps_decay_steps
        )
        return (
            self.cfg.eps_start
            + frac
            * (
                self.cfg.eps_end
                - self.cfg.eps_start
            )
        )

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        greedy: bool = False,
    ) -> int:
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                "obs must have shape "
                f"({self.obs_dim},), got {obs.shape}"
            )

        if (
            not greedy
            and self.rng.random() < self.epsilon()
        ):
            return int(
                self.rng.integers(
                    0,
                    self.n_actions,
                )
            )

        x = (
            torch.from_numpy(
                obs.astype(np.float32)
            )
            .to(self.device)
            .unsqueeze(0)
        )
        q_values = self.q(x)[0]
        return int(
            q_values.argmax().detach().cpu().item()
        )

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        self.rb.push(
            s,
            a,
            r,
            s2,
            done,
        )

    def update(
        self,
    ) -> Dict[str, float]:
        self.step += 1

        if len(self.rb) < max(
            self.cfg.min_replay,
            self.cfg.batch_size,
        ):
            return {
                "loss": float("nan"),
                "eps": float(self.epsilon()),
            }

        s, a, r, s2, d = self.rb.sample(
            self.cfg.batch_size
        )

        s_t = torch.from_numpy(s).to(
            self.device
        )
        a_t = torch.from_numpy(a).to(
            self.device
        )
        r_t = torch.from_numpy(r).to(
            self.device
        )
        s2_t = torch.from_numpy(s2).to(
            self.device
        )
        d_t = torch.from_numpy(d).to(
            self.device
        )

        q_sa = (
            self.q(s_t)
            .gather(
                1,
                a_t.view(-1, 1),
            )
            .squeeze(1)
        )

        with torch.no_grad():
            q_next = self.q_tgt(
                s2_t
            ).max(1)[0]
            target = (
                r_t
                + self.cfg.gamma
                * (1.0 - d_t)
                * q_next
            )

        loss = nn.functional.mse_loss(
            q_sa,
            target,
        )

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q.parameters(),
            self.cfg.grad_clip_norm,
        )
        self.optim.step()

        if (
            self.step
            % self.cfg.target_update
            == 0
        ):
            self.q_tgt.load_state_dict(
                self.q.state_dict()
            )

        return {
            "loss": float(
                loss.detach().cpu().item()
            ),
            "eps": float(self.epsilon()),
        }

    def save(
        self,
        path: str | Path,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            {
                "state_dict": self.q.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "seed": self.seed,
                "config": asdict(self.cfg),
            },
            destination,
        )

    def load(
        self,
        path: str | Path,
    ) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
        )

        checkpoint_obs_dim = checkpoint.get(
            "obs_dim"
        )
        checkpoint_actions = checkpoint.get(
            "n_actions"
        )

        if (
            checkpoint_obs_dim is not None
            and int(checkpoint_obs_dim)
            != self.obs_dim
        ):
            raise ValueError(
                "Checkpoint observation dimension "
                f"{checkpoint_obs_dim} does not match "
                f"environment dimension {self.obs_dim}."
            )
        if (
            checkpoint_actions is not None
            and int(checkpoint_actions)
            != self.n_actions
        ):
            raise ValueError(
                "Checkpoint action count "
                f"{checkpoint_actions} does not match "
                f"environment action count {self.n_actions}."
            )

        self.q.load_state_dict(
            checkpoint["state_dict"]
        )
        self.q_tgt.load_state_dict(
            self.q.state_dict()
        )
