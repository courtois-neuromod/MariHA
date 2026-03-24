"""Online continual learning With Likelihood adjustment (OWL).

Extends EWC regularisation with a UCB1 bandit that adaptively scales the
regularisation coefficient per task.  Tasks with higher UCB scores receive
stronger regularisation, biasing the update toward protecting the weights
most important for those tasks.

The bandit is updated at each task boundary with the cumulative return from
that task (as a proxy for forgetting risk).  When no test environments are
available (MariHA default), the bandit weight is used purely during training.

Reference: arXiv:1906.00322 (OWL-inspired; see also Evron et al., 2022).
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import tensorflow as tf

from mariha.methods.ewc import EWC_SAC


# ---------------------------------------------------------------------------
# UCB1 bandit
# ---------------------------------------------------------------------------


class UCBBandit:
    """UCB1 bandit over ``num_arms`` tasks.

    Args:
        num_arms: Number of task arms.
        c: Exploration constant.  Higher values favour arms seen less often.
    """

    def __init__(self, num_arms: int, c: float = 1.0) -> None:
        self.num_arms = num_arms
        self.c = c
        self.counts = np.zeros(num_arms, dtype=np.float64)
        self.values = np.zeros(num_arms, dtype=np.float64)
        self.total = 0

    def update(self, arm: int, reward: float) -> None:
        """Update arm estimate with an observed reward."""
        self.counts[arm] += 1
        self.total += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def ucb_scores(self) -> np.ndarray:
        """Return UCB1 scores, normalised to sum to 1."""
        if self.total == 0:
            return np.ones(self.num_arms) / self.num_arms
        scores = np.empty(self.num_arms)
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                scores[i] = float("inf")
            else:
                scores[i] = self.values[i] + self.c * math.sqrt(
                    math.log(self.total) / self.counts[i]
                )
        scores = np.clip(scores, 0.0, None)
        total = scores.sum()
        return scores / total if total > 0 else np.ones(self.num_arms) / self.num_arms

    def best_arm(self) -> int:
        """Return the arm with the highest UCB score."""
        return int(np.argmax(self.ucb_scores()))


# ---------------------------------------------------------------------------
# OWL_SAC
# ---------------------------------------------------------------------------


class OWL_SAC(EWC_SAC):
    """OWL: EWC regularisation weighted by a UCB1 bandit.

    At each task boundary the bandit weight for the completed task is updated.
    Before the next task begins, the EWC regularisation coefficient is scaled
    by the UCB score of the *new* task, stored in a ``tf.Variable`` so it can
    be read inside a compiled ``tf.function`` without re-tracing.

    Args:
        cl_reg_coef: Base EWC regularisation coefficient λ.
        regularize_critic: Also regularise critic weights.
        bandit_c: UCB1 exploration constant.
        **kwargs: Forwarded to :class:`~mariha.methods.ewc.EWC_SAC`.
    """

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        regularize_critic: bool = False,
        bandit_c: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            cl_reg_coef=cl_reg_coef,
            regularize_critic=regularize_critic,
            **kwargs,
        )
        self.bandit = UCBBandit(self.num_tasks, c=bandit_c)
        # Stores current task's normalised bandit weight as a tf.Variable so
        # it can be read inside a @tf.function without triggering a retrace.
        self._bandit_scale = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self._task_return_acc = 0.0

    # ------------------------------------------------------------------
    # Override auxiliary loss to scale by bandit weight
    # ------------------------------------------------------------------

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        """EWC regularisation loss scaled by the bandit weight."""
        base_loss = super().get_auxiliary_loss(seq_idx)
        return base_loss * self._bandit_scale

    # ------------------------------------------------------------------
    # SAC lifecycle hooks
    # ------------------------------------------------------------------

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        self._task_return_acc = 0.0
        # Scale the regularisation by the UCB score for this task.
        # Multiply by num_tasks so the average scale across tasks is ~1.
        scores = self.bandit.ucb_scores()
        w = float(scores[current_task_idx]) * self.num_tasks
        self._bandit_scale.assign(w)

    def on_task_end(self, current_task_idx: int) -> None:
        super().on_task_end(current_task_idx)
        self.bandit.update(current_task_idx, self._task_return_acc)

    def get_bandit_task(self) -> int:
        """Select the task with the highest UCB score (for evaluation)."""
        return self.bandit.best_arm()
