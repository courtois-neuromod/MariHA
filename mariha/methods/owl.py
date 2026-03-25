"""Online continual learning Without confLict (OWL).

Implements the continual RL method from Kessler et al.:
  "Same State, Different Task: Continual Reinforcement Learning without Interference"
  AAAI 2022 — arXiv:2106.02940

OWL prevents catastrophic forgetting through three mechanisms:

1. **Multi-head architecture** — one task-specific output head per task; the
   shared trunk never sees the task one-hot (``hide_task_id=True``).  With
   ``num_heads > 1``, ``common_variables`` returns only the trunk, so EWC
   automatically regularises the shared features without penalising the
   task-specific heads.

2. **EWC regularisation on the shared trunk** — same Fisher-diagonal
   importance weights as plain EWC, restricted to the trunk via
   ``common_variables``.

3. **EWAF bandit** (Exponentially Weighted Average Forecaster) — maintains
   a probability distribution over task heads.  At each task boundary the
   distribution is updated with the mean TD error from that task; the head
   with the highest probability can be used at test-time when the task
   identity is unknown.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import tensorflow as tf

from mariha.methods.ewc import EWC_SAC
from mariha.utils.running import create_one_hot_vec


# ---------------------------------------------------------------------------
# EWAF bandit
# ---------------------------------------------------------------------------


class EWAFBandit:
    """Exponentially Weighted Average Forecaster (hedge variant).

    Maintains a distribution over ``num_arms`` arms using importance-weighted
    exponential updates.  A lower loss for an arm increases its probability.

    Args:
        num_arms: Number of arms (task heads).
        eta: Learning rate.  Higher values make the distribution more peaked
            toward recently low-loss arms.
    """

    def __init__(self, num_arms: int, eta: float = 0.1) -> None:
        self.num_arms = num_arms
        self.eta = eta
        # Log-weights; initialised uniformly (all zeros → uniform softmax).
        self._log_w = np.zeros(num_arms, dtype=np.float64)

    def probabilities(self) -> np.ndarray:
        """Return a softmax probability vector over arms."""
        shifted = self._log_w - self._log_w.max()  # numerical stability
        p = np.exp(shifted)
        return p / p.sum()

    def best_arm(self) -> int:
        """Return the arm with the highest probability."""
        return int(np.argmax(self._log_w))

    def update(self, arm: int, loss: float) -> None:
        """Importance-weighted update for the selected arm.

        Args:
            arm: Index of the arm that was evaluated.
            loss: Non-negative loss signal.  Lower loss increases the arm's
                probability (``log_w[arm] -= eta * loss / p[arm]``).
        """
        p = self.probabilities()
        if p[arm] > 0.0:
            self._log_w[arm] -= self.eta * loss / p[arm]

    def reset(self) -> None:
        """Reset all log-weights to uniform."""
        self._log_w = np.zeros(self.num_arms, dtype=np.float64)


# ---------------------------------------------------------------------------
# OWL_SAC
# ---------------------------------------------------------------------------


class OWL_SAC(EWC_SAC):
    """OWL: multi-head SAC + EWC on shared trunk + EWAF bandit.

    Forces the multi-head architecture required by OWL:

    - ``num_heads`` is set to ``num_tasks`` so each task gets its own output
      head.
    - ``hide_task_id`` is set to ``True`` so the shared trunk processes
      observations without access to the task identity.
    - ``reset_buffer_on_task_change`` is ``True`` (paper default: flush
      task-specific data at each boundary).

    With ``num_heads > 1``, ``actor.common_variables`` and
    ``critic.common_variables`` return only the trunk parameters, so EWC
    automatically restricts its penalty to the shared features.

    The EWAF bandit is updated at the end of each task using the mean TD
    error accumulated during training on that task.  Call
    :meth:`get_bandit_action` to select an action using the bandit-inferred
    head at test-time when the true task identity is unknown.

    Args:
        cl_reg_coef: EWC regularisation coefficient λ.
        regularize_critic: Also regularise critic trunk weights.
        ewaf_eta: EWAF bandit learning rate η.
        **kwargs: Forwarded to :class:`~mariha.methods.ewc.EWC_SAC`.
    """

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        regularize_critic: bool = False,
        ewaf_eta: float = 0.1,
        **kwargs,
    ) -> None:
        # Force multi-head setup.  scene_ids must be present in kwargs (it
        # is always supplied as a keyword argument by from_args/run_benchmark).
        scene_ids = kwargs.get("scene_ids", [])
        num_tasks = len(scene_ids)

        policy_kwargs = dict(kwargs.pop("policy_kwargs", None) or {})
        policy_kwargs["num_heads"] = num_tasks
        policy_kwargs["hide_task_id"] = True
        kwargs["policy_kwargs"] = policy_kwargs
        # OWL flushes the replay buffer at each task switch (paper default).
        kwargs["reset_buffer_on_task_change"] = True

        super().__init__(
            cl_reg_coef=cl_reg_coef,
            regularize_critic=regularize_critic,
            **kwargs,
        )

        self.ewaf = EWAFBandit(self.num_tasks, eta=ewaf_eta)
        self._task_td_errors: List[float] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        self._task_td_errors = []

    def on_task_end(self, current_task_idx: int) -> None:
        """Update the EWAF with mean TD error accumulated during this task."""
        super().on_task_end(current_task_idx)
        if self._task_td_errors:
            mean_td = float(np.mean(self._task_td_errors))
            self.ewaf.update(current_task_idx, mean_td)

    # ------------------------------------------------------------------
    # TD-error collection
    # ------------------------------------------------------------------

    def _log_after_update(self, results: Dict) -> None:
        super()._log_after_update(results)
        if "abs_error" in results:
            self._task_td_errors.append(
                float(tf.reduce_mean(results["abs_error"]))
            )

    # ------------------------------------------------------------------
    # Bandit-based action selection (task-agnostic inference)
    # ------------------------------------------------------------------

    def get_bandit_action(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> int:
        """Select an action using the EWAF-inferred task head.

        Uses the task head with the highest EWAF probability, useful at
        test-time when the true task identity is unknown.

        Args:
            obs: Current observation.
            deterministic: If ``True``, return the greedy action.

        Returns:
            Integer action.
        """
        arm = self.ewaf.best_arm()
        one_hot = create_one_hot_vec(self.num_tasks, arm)
        return self.get_action_numpy(obs, one_hot, deterministic=deterministic)
