"""RL environment: pathway steering with delayed drug effects and pairwise drug interactions.

This environment keeps the original RL-Path idea, but makes the transition closer to a
biological treatment process:
- State core: pathway activity vector in [0, 1]^P
- Action: choose one drug index in [0, N_drugs)
- Delayed effects: a drug can act over several future steps via a temporal kernel
- Drug-drug interactions: the current drug can have synergy/antagonism with recently given drugs

To preserve the Markov property for DQN, the observation includes:
- current pathway activity vector
- summary of pending delayed effects still scheduled to land in future steps
- one-hot encoding of the most recent action (or zeros at episode start)
- normalized remaining-step fraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


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
        include_last_action: bool = True,
        include_pending_summary: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        effects
            Drug→pathway matrix in [0,1], shape (N_drugs, N_pathways).
        alpha
            Strength of intervention effect.
        step_penalty
            Constant penalty each step (discourages long sequences).
        action_cost_scale
            Penalty proportional to action breadth.
        noise
            Small Gaussian noise in transitions.
        disease_pathway_frac
            Fraction of pathways treated as disease-high at reset.
        temporal_kernel
            Fractions of a drug effect applied across current and future steps.
            Example [0.5, 0.3, 0.2] means 50% now, 30% next step, 20% the step after.
        interaction_scale
            If interaction_matrix is not supplied, build a simple symmetric heuristic from
            overlap in pathway coverage and scale it by this value. Positive = synergy.
            Negative = antagonism.
        interaction_matrix
            Optional explicit pairwise interaction tensor with shape
            (N_drugs, N_drugs, N_pathways). Entry [i, j, p] is the extra pathway effect on
            pathway p when action j follows prior drug i.
        interaction_history
            How many recent actions contribute pairwise interactions.
        """
        self.effects = np.asarray(effects, dtype=np.float32)
        if self.effects.ndim != 2:
            raise ValueError("effects must have shape (N_drugs, N_pathways)")

        self.drug_names = list(drug_names)
        self.pathway_names = list(pathway_names)
        self.n_actions, self.n_pathways = self.effects.shape
        if len(self.drug_names) != self.n_actions:
            raise ValueError("drug_names length must match effects.shape[0]")
        if len(self.pathway_names) != self.n_pathways:
            raise ValueError("pathway_names length must match effects.shape[1]")

        self.max_steps = int(steps)
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.step_penalty = float(step_penalty)
        self.action_cost_scale = float(action_cost_scale)
        self.noise = float(noise)
        self.disease_pathway_frac = float(disease_pathway_frac)
        self.interaction_history = max(1, int(interaction_history))
        self.include_last_action = bool(include_last_action)
        self.include_pending_summary = bool(include_pending_summary)

        self.temporal_kernel = self._normalize_temporal_kernel(temporal_kernel)
        self.pending_effects: List[Dict[str, np.ndarray | int]] = []
        self.action_history: List[int] = []
        self.last_action: Optional[int] = None

        self.interaction_matrix = self._init_interaction_matrix(
            interaction_matrix=interaction_matrix,
            interaction_scale=float(interaction_scale),
        )

        self.disease_mask = self._make_disease_mask()
        self.target = np.zeros(self.n_pathways, dtype=np.float32)

        self.t = 0
        self.state = np.zeros(self.n_pathways, dtype=np.float32)

        self.obs_dim = self.n_pathways
        if self.include_pending_summary:
            self.obs_dim += self.n_pathways
        if self.include_last_action:
            self.obs_dim += self.n_actions
        self.obs_dim += 1  # normalized remaining-step fraction

    @staticmethod
    def _normalize_temporal_kernel(kernel: Optional[Sequence[float]]) -> np.ndarray:
        if kernel is None:
            return np.array([1.0], dtype=np.float32)
        arr = np.asarray(list(kernel), dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("temporal_kernel must be a non-empty 1D sequence")
        if np.any(arr < 0):
            raise ValueError("temporal_kernel values must be non-negative")
        total = float(arr.sum())
        if total <= 0.0:
            raise ValueError("temporal_kernel must sum to a positive value")
        return arr / total

    def _init_interaction_matrix(
        self,
        interaction_matrix: Optional[np.ndarray],
        interaction_scale: float,
    ) -> np.ndarray:
        if interaction_matrix is not None:
            mat = np.asarray(interaction_matrix, dtype=np.float32)
            expected = (self.n_actions, self.n_actions, self.n_pathways)
            if mat.shape != expected:
                raise ValueError(f"interaction_matrix must have shape {expected}, got {mat.shape}")
            return mat

        if interaction_scale == 0.0:
            return np.zeros((self.n_actions, self.n_actions, self.n_pathways), dtype=np.float32)

        # Heuristic pairwise interaction: pathway-level overlap between prior and current drug.
        # This yields a simple symmetric synergy tensor. Negative interaction_scale turns it into
        # antagonism.
        overlap = np.minimum(self.effects[:, None, :], self.effects[None, :, :])
        return (interaction_scale * overlap).astype(np.float32)

    def _make_disease_mask(self) -> np.ndarray:
        k = max(1, int(round(self.n_pathways * self.disease_pathway_frac)))
        idx = self.rng.choice(self.n_pathways, size=k, replace=False)
        mask = np.zeros(self.n_pathways, dtype=bool)
        mask[idx] = True
        return mask

    def _pending_summary(self) -> np.ndarray:
        summary = np.zeros(self.n_pathways, dtype=np.float32)
        for item in self.pending_effects:
            effect = np.asarray(item["effect"], dtype=np.float32)
            delay = int(item["delay"])
            weight = 1.0 / float(delay + 1)
            summary += weight * effect
        return np.clip(summary, 0.0, 1.0).astype(np.float32)

    def _last_action_one_hot(self) -> np.ndarray:
        vec = np.zeros(self.n_actions, dtype=np.float32)
        if self.last_action is not None:
            vec[self.last_action] = 1.0
        return vec

    def _remaining_fraction(self) -> np.ndarray:
        remaining = max(self.max_steps - self.t, 0)
        frac = remaining / float(max(self.max_steps, 1))
        return np.array([frac], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        chunks = [self.state.astype(np.float32, copy=True)]
        if self.include_pending_summary:
            chunks.append(self._pending_summary())
        if self.include_last_action:
            chunks.append(self._last_action_one_hot())
        chunks.append(self._remaining_fraction())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.pending_effects = []
        self.action_history = []
        self.last_action = None

        s = self.rng.uniform(0.25, 0.55, size=self.n_pathways).astype(np.float32)
        s[self.disease_mask] = self.rng.uniform(
            0.65,
            0.95,
            size=int(self.disease_mask.sum()),
        ).astype(np.float32)
        self.state = s
        return self._get_obs()

    def _apply_due_effects(self) -> Tuple[np.ndarray, List[Dict[str, np.ndarray | int]]]:
        total = np.zeros(self.n_pathways, dtype=np.float32)
        still_pending: List[Dict[str, np.ndarray | int]] = []
        due_items: List[Dict[str, np.ndarray | int]] = []
        for item in self.pending_effects:
            next_delay = int(item["delay"]) - 1
            if next_delay <= 0:
                eff = np.asarray(item["effect"], dtype=np.float32)
                total += eff
                due_items.append({"delay": 0, "effect": eff})
            else:
                still_pending.append({"delay": next_delay, "effect": np.asarray(item["effect"], dtype=np.float32)})
        self.pending_effects = still_pending
        return total.astype(np.float32), due_items

    def _schedule_effect(self, effect: np.ndarray) -> None:
        for delay, weight in enumerate(self.temporal_kernel):
            piece = (weight * effect).astype(np.float32)
            if delay == 0:
                self.pending_effects.append({"delay": 0, "effect": piece})
            else:
                self.pending_effects.append({"delay": int(delay), "effect": piece})

    def _pairwise_interaction_effect(self, action: int) -> np.ndarray:
        if self.interaction_matrix is None or not self.action_history:
            return np.zeros(self.n_pathways, dtype=np.float32)
        total = np.zeros(self.n_pathways, dtype=np.float32)
        recent = self.action_history[-self.interaction_history :]
        for prev_action in recent:
            total += self.interaction_matrix[prev_action, action]
        return total.astype(np.float32)

    def compute_action_components(self, action: int) -> Dict[str, np.ndarray | float]:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action out of range: {action}")
        base = self.effects[action].astype(np.float32)
        interaction = self._pairwise_interaction_effect(action)
        total_effect = np.clip(base + interaction, 0.0, 1.0).astype(np.float32)
        action_cost = float(self.action_cost_scale * (base.sum() / (self.n_pathways + 1e-6)))
        return {
            "base_effect": base,
            "interaction_effect": interaction,
            "scheduled_effect": total_effect,
            "action_cost": action_cost,
        }

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action out of range: {action}")

        self.t += 1
        s = self.state.astype(np.float32, copy=True)
        prev_dist = float(np.mean((s[self.disease_mask] - self.target[self.disease_mask]) ** 2))

        action_parts = self.compute_action_components(action)
        self._schedule_effect(np.asarray(action_parts["scheduled_effect"], dtype=np.float32))
        due_effect, due_items = self._apply_due_effects()

        eps = self.rng.normal(0.0, self.noise, size=self.n_pathways).astype(np.float32)
        delta = -self.alpha * due_effect
        s2 = np.clip(s + delta + eps, 0.0, 1.0).astype(np.float32)

        new_dist = float(np.mean((s2[self.disease_mask] - self.target[self.disease_mask]) ** 2))
        improvement = prev_dist - new_dist
        reward = float(improvement - self.step_penalty - float(action_parts["action_cost"]))

        self.state = s2
        self.last_action = int(action)
        self.action_history.append(int(action))
        done = self.t >= self.max_steps

        info = {
            "t": self.t,
            "drug": self.drug_names[action],
            "prev_disease_mse": prev_dist,
            "new_disease_mse": new_dist,
            "improvement": improvement,
            "action_cost": float(action_parts["action_cost"]),
            "base_effect_sum": float(np.sum(action_parts["base_effect"])),
            "interaction_effect_sum": float(np.sum(action_parts["interaction_effect"])),
            "applied_effect_sum": float(np.sum(due_effect)),
            "pending_effect_sum": float(np.sum(self._pending_summary())),
            "n_due_effect_chunks": len(due_items),
            "last_action": self.last_action,
        }
        return StepResult(obs=self._get_obs(), reward=reward, done=done, info=info)

    def sample_action(self) -> int:
        return int(self.rng.integers(0, self.n_actions))
