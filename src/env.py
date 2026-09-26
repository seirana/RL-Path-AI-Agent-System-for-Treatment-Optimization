"""Pathway-steering reinforcement-learning environment.

The environment is a research simulation. Drug/pathway effects, delayed dynamics, and
pairwise interactions are modeling abstractions and are not validated treatment effects.

State/action/reward definition
------------------------------
State
    Current pathway activity, disease-pathway mask, exact pending delayed-effect schedule,
    recent action history, and remaining episode fraction.
Action
    One discrete drug index.
Reward
    Reduction in mean-squared disease-pathway activity minus a per-step penalty and an
    action-breadth cost.

The observation exposes every variable used by the transition/reward logic so the default
configuration is Markov with respect to the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class PathwaySteeringEnv:
    def __init__(
        self,
        effects: np.ndarray,
        drug_names: List[str],
        pathway_names: List[str],
        steps: int = 10,
        seed: int = 42,
        alpha: float = 0.8,
        step_penalty: float = 0.02,
        action_cost_scale: float = 0.05,
        noise: float = 0.01,
        disease_pathway_frac: float = 0.35,
        temporal_kernel: Optional[Sequence[float]] = None,
        interaction_scale: float = 0.0,
        interaction_matrix: Optional[np.ndarray] = None,
        interaction_history: int = 1,
        disease_mask: Optional[np.ndarray] = None,
        include_disease_mask: bool = True,
        include_action_history: bool = True,
        include_pending_schedule: bool = True,
    ) -> None:
        self.effects = np.asarray(effects, dtype=np.float32)
        if self.effects.ndim != 2:
            raise ValueError("effects must have shape (N_drugs, N_pathways)")
        if self.effects.shape[0] == 0 or self.effects.shape[1] == 0:
            raise ValueError("effects must contain at least one drug and pathway")
        if not np.all(np.isfinite(self.effects)):
            raise ValueError("effects must contain only finite values")
        if np.any(self.effects < 0.0) or np.any(self.effects > 1.0):
            raise ValueError("effects values must be in [0, 1]")

        self.drug_names = list(drug_names)
        self.pathway_names = list(pathway_names)
        self.n_actions, self.n_pathways = self.effects.shape
        if len(self.drug_names) != self.n_actions:
            raise ValueError("drug_names length must match effects.shape[0]")
        if len(self.pathway_names) != self.n_pathways:
            raise ValueError("pathway_names length must match effects.shape[1]")

        self.max_steps = int(steps)
        if self.max_steps <= 0:
            raise ValueError("steps must be greater than 0")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if action_cost_scale < 0:
            raise ValueError("action_cost_scale must be non-negative")
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if not 0.0 < disease_pathway_frac <= 1.0:
            raise ValueError("disease_pathway_frac must be in (0, 1]")
        if int(interaction_history) <= 0:
            raise ValueError("interaction_history must be greater than 0")

        self.seed = int(seed)
        self._set_rngs(self.seed)

        self.alpha = float(alpha)
        self.step_penalty = float(step_penalty)
        self.action_cost_scale = float(action_cost_scale)
        self.noise = float(noise)
        self.disease_pathway_frac = float(disease_pathway_frac)
        self.interaction_history = int(interaction_history)

        self.include_disease_mask = bool(include_disease_mask)
        self.include_action_history = bool(include_action_history)
        self.include_pending_schedule = bool(include_pending_schedule)

        self.temporal_kernel = self._normalize_temporal_kernel(temporal_kernel)
        self.pending_horizon = max(int(self.temporal_kernel.size) - 1, 0)
        self.pending_effects: List[Dict[str, np.ndarray | int]] = []
        self.action_history: List[int] = []
        self.last_action: Optional[int] = None

        self.interaction_matrix = self._init_interaction_matrix(
            interaction_matrix=interaction_matrix,
            interaction_scale=float(interaction_scale),
        )

        if disease_mask is None:
            self.disease_mask = self._make_disease_mask()
            self._fixed_disease_mask = False
        else:
            mask = np.asarray(disease_mask, dtype=bool)
            if mask.shape != (self.n_pathways,):
                raise ValueError(
                    "disease_mask must have shape "
                    f"({self.n_pathways},), got {mask.shape}"
                )
            if not np.any(mask):
                raise ValueError("disease_mask must select at least one pathway")
            self.disease_mask = mask.copy()
            self._fixed_disease_mask = True

        self.target = np.zeros(self.n_pathways, dtype=np.float32)
        self.t = 0
        self.state = np.zeros(self.n_pathways, dtype=np.float32)

        self.obs_dim = self.n_pathways
        if self.include_disease_mask:
            self.obs_dim += self.n_pathways
        if self.include_pending_schedule:
            self.obs_dim += self.pending_horizon * self.n_pathways
        if self.include_action_history:
            self.obs_dim += self.interaction_history * self.n_actions
        self.obs_dim += 1

    def _set_rngs(self, seed: int) -> None:
        """Use independent RNG streams for environment dynamics and random actions."""

        sequence = np.random.SeedSequence(int(seed))
        transition_seed, action_seed = sequence.spawn(2)
        self.rng = np.random.default_rng(transition_seed)
        self.action_rng = np.random.default_rng(action_seed)

    @staticmethod
    def _normalize_temporal_kernel(
        kernel: Optional[Sequence[float]],
    ) -> np.ndarray:
        if kernel is None:
            return np.array([1.0], dtype=np.float32)

        arr = np.asarray(list(kernel), dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("temporal_kernel must be a non-empty 1D sequence")
        if not np.all(np.isfinite(arr)):
            raise ValueError("temporal_kernel values must be finite")
        if np.any(arr < 0):
            raise ValueError("temporal_kernel values must be non-negative")

        total = float(arr.sum())
        if total <= 0.0:
            raise ValueError("temporal_kernel must sum to a positive value")
        return (arr / total).astype(np.float32)

    def _init_interaction_matrix(
        self,
        interaction_matrix: Optional[np.ndarray],
        interaction_scale: float,
    ) -> np.ndarray:
        if interaction_matrix is not None:
            mat = np.asarray(interaction_matrix, dtype=np.float32)
            expected = (
                self.n_actions,
                self.n_actions,
                self.n_pathways,
            )
            if mat.shape != expected:
                raise ValueError(
                    "interaction_matrix must have shape "
                    f"{expected}, got {mat.shape}"
                )
            if not np.all(np.isfinite(mat)):
                raise ValueError(
                    "interaction_matrix must contain only finite values"
                )
            return mat

        if interaction_scale == 0.0:
            return np.zeros(
                (
                    self.n_actions,
                    self.n_actions,
                    self.n_pathways,
                ),
                dtype=np.float32,
            )

        overlap = np.minimum(
            self.effects[:, None, :],
            self.effects[None, :, :],
        )
        return (interaction_scale * overlap).astype(np.float32)

    def _make_disease_mask(self) -> np.ndarray:
        k = max(
            1,
            int(round(self.n_pathways * self.disease_pathway_frac)),
        )
        idx = self.rng.choice(
            self.n_pathways,
            size=k,
            replace=False,
        )
        mask = np.zeros(self.n_pathways, dtype=bool)
        mask[idx] = True
        return mask

    def _pending_schedule_observation(self) -> np.ndarray:
        if self.pending_horizon == 0:
            return np.empty(0, dtype=np.float32)

        schedule = np.zeros(
            (self.pending_horizon, self.n_pathways),
            dtype=np.float32,
        )
        for item in self.pending_effects:
            delay = int(item["delay"])
            if 0 <= delay < self.pending_horizon:
                schedule[delay] += np.asarray(
                    item["effect"],
                    dtype=np.float32,
                )

        return schedule.reshape(-1).astype(np.float32)

    def _pending_summary(self) -> np.ndarray:
        summary = np.zeros(
            self.n_pathways,
            dtype=np.float32,
        )
        for item in self.pending_effects:
            summary += np.asarray(
                item["effect"],
                dtype=np.float32,
            )
        return summary

    def _action_history_observation(self) -> np.ndarray:
        history = np.zeros(
            (self.interaction_history, self.n_actions),
            dtype=np.float32,
        )
        recent = self.action_history[-self.interaction_history :]
        if recent:
            offset = self.interaction_history - len(recent)
            for index, action in enumerate(recent):
                history[offset + index, action] = 1.0
        return history.reshape(-1)

    def _remaining_fraction(self) -> np.ndarray:
        remaining = max(self.max_steps - self.t, 0)
        frac = remaining / float(self.max_steps)
        return np.array([frac], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        chunks = [
            self.state.astype(
                np.float32,
                copy=True,
            )
        ]
        if self.include_disease_mask:
            chunks.append(
                self.disease_mask.astype(np.float32)
            )
        if self.include_pending_schedule:
            chunks.append(
                self._pending_schedule_observation()
            )
        if self.include_action_history:
            chunks.append(
                self._action_history_observation()
            )
        chunks.append(self._remaining_fraction())
        return np.concatenate(chunks).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        resample_disease_mask: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
            self._set_rngs(self.seed)

        if resample_disease_mask:
            if self._fixed_disease_mask:
                raise ValueError(
                    "Cannot resample a user-supplied fixed disease_mask"
                )
            self.disease_mask = self._make_disease_mask()

        self.t = 0
        self.pending_effects = []
        self.action_history = []
        self.last_action = None

        if initial_state is not None:
            state = np.asarray(
                initial_state,
                dtype=np.float32,
            )
            if state.shape != (self.n_pathways,):
                raise ValueError(
                    "initial_state must have shape "
                    f"({self.n_pathways},), got {state.shape}"
                )
            if not np.all(np.isfinite(state)):
                raise ValueError(
                    "initial_state must contain only finite values"
                )
            self.state = np.clip(
                state,
                0.0,
                1.0,
            ).astype(np.float32)
            return self._get_obs()

        state = self.rng.uniform(
            0.25,
            0.55,
            size=self.n_pathways,
        ).astype(np.float32)
        state[self.disease_mask] = self.rng.uniform(
            0.65,
            0.95,
            size=int(self.disease_mask.sum()),
        ).astype(np.float32)
        self.state = state
        return self._get_obs()

    def _apply_due_effects(
        self,
    ) -> Tuple[
        np.ndarray,
        List[Dict[str, np.ndarray | int]],
    ]:
        """Apply delay=0 chunks and decrement only future chunks.

        The previous implementation decremented every delay before checking
        whether it was due, causing both delay=0 and delay=1 effects to land
        on the same transition. This version preserves the intended temporal
        kernel semantics.
        """

        total = np.zeros(
            self.n_pathways,
            dtype=np.float32,
        )
        still_pending: List[
            Dict[str, np.ndarray | int]
        ] = []
        due_items: List[
            Dict[str, np.ndarray | int]
        ] = []

        for item in self.pending_effects:
            delay = int(item["delay"])
            effect = np.asarray(
                item["effect"],
                dtype=np.float32,
            )

            if delay <= 0:
                total += effect
                due_items.append(
                    {
                        "delay": 0,
                        "effect": effect,
                    }
                )
            else:
                still_pending.append(
                    {
                        "delay": delay - 1,
                        "effect": effect,
                    }
                )

        self.pending_effects = still_pending
        return total.astype(np.float32), due_items

    def _schedule_effect(
        self,
        effect: np.ndarray,
    ) -> None:
        for delay, weight in enumerate(
            self.temporal_kernel
        ):
            if weight <= 0:
                continue
            piece = (
                float(weight) * effect
            ).astype(np.float32)
            self.pending_effects.append(
                {
                    "delay": int(delay),
                    "effect": piece,
                }
            )

    def _pairwise_interaction_effect(
        self,
        action: int,
    ) -> np.ndarray:
        if not self.action_history:
            return np.zeros(
                self.n_pathways,
                dtype=np.float32,
            )

        total = np.zeros(
            self.n_pathways,
            dtype=np.float32,
        )
        recent = self.action_history[
            -self.interaction_history :
        ]
        for prev_action in recent:
            total += self.interaction_matrix[
                prev_action,
                action,
            ]
        return total.astype(np.float32)

    def compute_action_components(
        self,
        action: int,
    ) -> Dict[str, np.ndarray | float]:
        if action < 0 or action >= self.n_actions:
            raise ValueError(
                f"action out of range: {action}"
            )

        base = self.effects[action].astype(
            np.float32
        )
        interaction = (
            self._pairwise_interaction_effect(
                action
            )
        )
        total_effect = np.clip(
            base + interaction,
            0.0,
            1.0,
        ).astype(np.float32)

        action_cost = float(
            self.action_cost_scale
            * (
                base.sum()
                / (self.n_pathways + 1e-6)
            )
        )

        return {
            "base_effect": base,
            "interaction_effect": interaction,
            "scheduled_effect": total_effect,
            "action_cost": action_cost,
        }

    def disease_mse(self) -> float:
        return float(
            np.mean(
                (
                    self.state[self.disease_mask]
                    - self.target[self.disease_mask]
                )
                ** 2
            )
        )

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError(
                f"action out of range: {action}"
            )
        if self.t >= self.max_steps:
            raise RuntimeError(
                "Episode is already complete; call reset() before step()."
            )

        self.t += 1
        state = self.state.astype(
            np.float32,
            copy=True,
        )
        prev_dist = self.disease_mse()

        action_parts = (
            self.compute_action_components(action)
        )
        self._schedule_effect(
            np.asarray(
                action_parts["scheduled_effect"],
                dtype=np.float32,
            )
        )
        due_effect, due_items = (
            self._apply_due_effects()
        )

        eps = self.rng.normal(
            0.0,
            self.noise,
            size=self.n_pathways,
        ).astype(np.float32)
        delta = -self.alpha * due_effect
        next_state = np.clip(
            state + delta + eps,
            0.0,
            1.0,
        ).astype(np.float32)

        self.state = next_state
        new_dist = self.disease_mse()
        improvement = prev_dist - new_dist
        reward = float(
            improvement
            - self.step_penalty
            - float(action_parts["action_cost"])
        )

        self.last_action = int(action)
        self.action_history.append(int(action))
        done = self.t >= self.max_steps

        info: Dict[str, object] = {
            "t": self.t,
            "drug": self.drug_names[action],
            "prev_disease_mse": prev_dist,
            "new_disease_mse": new_dist,
            "improvement": improvement,
            "action_cost": float(
                action_parts["action_cost"]
            ),
            "base_effect_sum": float(
                np.sum(
                    action_parts["base_effect"]
                )
            ),
            "interaction_effect_sum": float(
                np.sum(
                    action_parts[
                        "interaction_effect"
                    ]
                )
            ),
            "applied_effect_sum": float(
                np.sum(due_effect)
            ),
            "pending_effect_sum": float(
                np.sum(self._pending_summary())
            ),
            "n_due_effect_chunks": len(
                due_items
            ),
            "last_action": self.last_action,
        }

        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            done=done,
            info=info,
        )

    def sample_action(self) -> int:
        return int(
            self.action_rng.integers(
                0,
                self.n_actions,
            )
        )
