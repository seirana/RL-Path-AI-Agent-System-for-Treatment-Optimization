"""Minimal DQN (Deep Q-Network) for discrete-action environments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
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


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buf.append((s.astype(np.float32), int(a), float(r), s2.astype(np.float32), bool(done)))

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int):
        if batch_size > len(self.buf):
            raise ValueError(f"batch_size={batch_size} exceeds replay size={len(self.buf)}")
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, seed: int = 0):
        self.cfg = cfg
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(obs_dim, n_actions, hidden=cfg.hidden_dim).to(self.device)
        self.q_tgt = QNet(obs_dim, n_actions, hidden=cfg.hidden_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.replay_size)
        self.n_actions = n_actions
        self.step = 0

    def epsilon(self) -> float:
        t = min(self.step, self.cfg.eps_decay_steps)
        frac = t / float(max(self.cfg.eps_decay_steps, 1))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (np.random.rand() < self.epsilon()):
            return int(np.random.randint(0, self.n_actions))
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device).unsqueeze(0)
        qv = self.q(x)[0].detach().cpu().numpy()
        return int(qv.argmax())

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.rb.push(s, a, r, s2, done)

    def update(self) -> Dict[str, float]:
        self.step += 1
        if len(self.rb) < max(self.cfg.min_replay, self.cfg.batch_size):
            return {"loss": float("nan"), "eps": float(self.epsilon())}

        s, a, r, s2, d = self.rb.sample(self.cfg.batch_size)

        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        s2_t = torch.from_numpy(s2).to(self.device)
        d_t = torch.from_numpy(d).to(self.device)

        q_sa = self.q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_tgt(s2_t).max(1)[0]
            target = r_t + self.cfg.gamma * (1.0 - d_t) * q_next

        loss = nn.functional.mse_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        if self.step % self.cfg.target_update == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.detach().cpu().item()), "eps": float(self.epsilon())}

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.q.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["state_dict"])
        self.q_tgt.load_state_dict(self.q.state_dict())
