"""Elastic Weight Consolidation (EWC) continual learning baseline.

Estimates parameter importance via the diagonal of the Fisher information
matrix, computed from gradients of log-policy and Q-value outputs.

Reference: Kirkpatrick et al., 2017 — arXiv:1612.00796.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf

from mariha.methods.regularization import Regularization_SAC


class EWC_SAC(Regularization_SAC):
    """EWC regularisation method.

    Args:
        cl_reg_coef: Regularisation coefficient λ.
        regularize_critic: Also regularise critic weights.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    @tf.function
    def _get_grads(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        """Compute Jacobians of the summed policy/Q outputs w.r.t. parameters."""
        with tf.GradientTape(persistent=True) as g:
            logits = tf.reduce_sum(self.actor(obs, one_hot), axis=-1)
            q1 = tf.reduce_sum(self.critic1(obs, one_hot), axis=-1)
            q2 = tf.reduce_sum(self.critic2(obs, one_hot), axis=-1)

        actor_gs = g.jacobian(logits, self.actor_common_variables)
        q1_gs = g.jacobian(q1, self.critic1.common_variables)
        q2_gs = g.jacobian(q2, self.critic2.common_variables)
        del g
        return actor_gs, q1_gs, q2_gs

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        actor_gs, q1_gs, q2_gs = self._get_grads(batch["obs"], batch["one_hot"])

        reg_weights = []
        for gs in actor_gs:
            if gs is None:
                raise ValueError("Actor Jacobian is None — check model architecture.")
            # Fisher diagonal: sum over output dim, clip and average over batch.
            fisher = tf.reduce_sum(gs ** 2, axis=1)
            fisher = tf.clip_by_value(fisher, 1e-5, np.inf)
            reg_weights.append(tf.reduce_mean(fisher, axis=0))

        critic_coef = 1.0 if self.regularize_critic else 0.0
        for gs in q1_gs:
            reg_weights.append(critic_coef * tf.reduce_mean(gs ** 2, axis=0))
        for gs in q2_gs:
            reg_weights.append(critic_coef * tf.reduce_mean(gs ** 2, axis=0))

        return reg_weights
