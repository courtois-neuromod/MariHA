"""Memory Aware Synapses (MAS) continual learning baseline.

Estimates parameter importance as the sensitivity of the network's output
magnitude to perturbations — computed as the gradient of the squared L2
norm of the outputs.

Reference: Aljundi et al., 2018 — arXiv:1711.09601.
"""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf

from mariha.methods.regularization import Regularization_SAC


class MAS_SAC(Regularization_SAC):
    """MAS regularisation method.

    Args:
        cl_reg_coef: Regularisation coefficient λ.
        regularize_critic: Also regularise critic weights.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    @tf.function
    def _get_grads(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        """Compute Jacobians of output squared-norm w.r.t. parameters."""
        with tf.GradientTape(persistent=True) as g:
            actor_norm = tf.reduce_sum(self.actor(obs, one_hot) ** 2, axis=-1)
            q1_norm = tf.reduce_sum(self.critic1(obs, one_hot) ** 2, axis=-1)
            q2_norm = tf.reduce_sum(self.critic2(obs, one_hot) ** 2, axis=-1)

        actor_gs = g.jacobian(actor_norm, self.actor_common_variables)
        q1_gs = g.jacobian(q1_norm, self.critic1.common_variables)
        q2_gs = g.jacobian(q2_norm, self.critic2.common_variables)
        del g
        return actor_gs, q1_gs, q2_gs

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        actor_gs, q1_gs, q2_gs = self._get_grads(batch["obs"], batch["one_hot"])

        reg_weights = []
        for gs in actor_gs:
            reg_weights.append(tf.reduce_mean(tf.abs(gs), axis=0))

        critic_coef = 1.0 if self.regularize_critic else 0.0
        for gs in q1_gs:
            reg_weights.append(critic_coef * tf.reduce_mean(tf.abs(gs), axis=0))
        for gs in q2_gs:
            reg_weights.append(critic_coef * tf.reduce_mean(tf.abs(gs), axis=0))

        return reg_weights
