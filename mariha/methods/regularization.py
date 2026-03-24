"""Base class for parameter-regularisation CL methods (L2, EWC, MAS, SI).

``Regularization_SAC`` extends ``SAC`` with three shared mechanisms:

1. **Old-parameter snapshots** (``old_params``): copied from the current
   network at each task boundary.
2. **Importance weights** (``reg_weights``): per-parameter scalars
   indicating how critical each weight is for previous tasks.
3. **Regularisation loss**: ``Σ_k weight_k · (θ_k − θ_k^old)^2``,
   added to both the actor and critic losses via ``get_auxiliary_loss()``.

Subclasses implement ``_get_importance_weights()`` to compute their
specific importance metric (Fisher diagonal for EWC, sensitivity for MAS,
uniform for L2, online surrogate for SI).

Ported from COOM; COOM-specific env references removed.
"""

from __future__ import annotations

from typing import List

import tensorflow as tf

from mariha.replay.buffers import ReplayBuffer
from mariha.rl.sac import SAC


class Regularization_SAC(SAC):
    """Base class for regularisation-based CL methods.

    Args:
        cl_reg_coef: Regularisation coefficient λ.  Scales the penalty term
            ``λ · Σ_k weight_k · (θ_k − θ_k^*)^2``.
        regularize_critic: If ``True``, regularise critic variables too.
            If ``False`` (default), only the actor is regularised.
        **vanilla_sac_kwargs: All remaining keyword arguments forwarded to
            :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        regularize_critic: bool = False,
        **vanilla_sac_kwargs,
    ) -> None:
        super().__init__(**vanilla_sac_kwargs)
        self.cl_reg_coef = cl_reg_coef
        self.regularize_critic = regularize_critic

        self.actor_common_variables = self.actor.common_variables
        self.critic_common_variables = (
            self.critic1.common_variables + self.critic2.common_variables
        )

        # Snapshot of parameter values from the end of the previous task.
        self.old_params = [
            tf.Variable(tf.identity(p), trainable=False)
            for p in self.all_common_variables
        ]

        # Per-parameter importance weights (accumulated across tasks).
        self.reg_weights = [
            tf.Variable(tf.zeros_like(p), trainable=False)
            for p in self.all_common_variables
        ]

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        """Return λ·reg_loss if we're past the first task, else 0."""
        reg_loss = self._regularize(self.old_params)
        coef = tf.cond(seq_idx > 0, lambda: self.cl_reg_coef, lambda: 0.0)
        return reg_loss * coef

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        if current_task_idx > 0:
            # Snapshot current parameters as the reference for this task.
            for old_p, cur_p in zip(self.old_params, self.all_common_variables):
                old_p.assign(cur_p)
            # Recompute importance weights from the buffer.
            self._update_reg_weights(self.replay_buffer, batch_size=32)

    def _merge_weights(self, new_weights: List[tf.Tensor]) -> None:
        """Accumulate importance weights: old + new."""
        merged = [o + n for o, n in zip(self.reg_weights, new_weights)]
        for old_w, merged_w in zip(self.reg_weights, merged):
            old_w.assign(merged_w)

    def _update_reg_weights(
        self,
        replay_buffer: ReplayBuffer,
        batches_num: int = 10,
        batch_size: int = 256,
    ) -> None:
        """Sample from the replay buffer and compute importance weights."""
        all_weights: List[List[tf.Tensor]] = []
        for _ in range(batches_num):
            batch = replay_buffer.sample_batch(min(batch_size, replay_buffer.size))
            all_weights.append(self._get_importance_weights(**batch))

        mean_weights = [
            tf.reduce_mean(tf.stack(ws, axis=0), axis=0)
            for ws in zip(*all_weights)
        ]
        self._merge_weights(mean_weights)

    def _regularize(self, old_params: List[tf.Variable]) -> tf.Tensor:
        """Compute Σ_k weight_k · (θ_k − θ_k^*)^2."""
        loss = tf.zeros([])
        for cur_p, old_p, w in zip(self.all_common_variables, old_params, self.reg_weights):
            loss += tf.reduce_sum(w * (cur_p - old_p) ** 2)
        return loss

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        """Subclasses must implement this to return a list of weight tensors."""
        raise NotImplementedError
