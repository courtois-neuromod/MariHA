"""Synaptic Intelligence (SI) continual learning baseline.

Unlike EWC and MAS which compute importance *after* a task ends by querying
the buffer, SI accumulates importance *online* during training by tracking
how much each parameter contributed to reducing the loss:

    ω_k += g_k · Δθ_k    (per gradient step)

At each task boundary the importance weights are consolidated into:

    Ω_k = ω_k / ((Δθ_k^total)^2 + ξ)

and the reference point θ_k^* is snapshotted.

Reference: Zenke et al., 2017 — arXiv:1703.04200.
"""

from __future__ import annotations

from typing import List, Optional

import tensorflow as tf

from mariha.rl.sac import SAC


class SI_SAC(SAC):
    """Synaptic Intelligence (online importance accumulation).

    Args:
        cl_reg_coef: Regularisation coefficient λ.
        regularize_critic: Also regularise critic weights.
        si_epsilon: Small constant ξ to avoid division by zero when
            consolidating importance weights.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        regularize_critic: bool = False,
        si_epsilon: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cl_reg_coef = cl_reg_coef
        self.regularize_critic = regularize_critic
        self.si_epsilon = si_epsilon

        self._tracked_vars = self._get_tracked_vars()

        # θ_k^* — reference params from the start of the current task.
        self._theta_star: List[tf.Variable] = [
            tf.Variable(tf.identity(v), trainable=False) for v in self._tracked_vars
        ]
        # θ_k^old — params at end of last task (used in regularisation loss).
        self._theta_old: List[tf.Variable] = [
            tf.Variable(tf.identity(v), trainable=False) for v in self._tracked_vars
        ]
        # ω_k — running surrogate gradient × delta-param products.
        self._small_omega: List[tf.Variable] = [
            tf.Variable(tf.zeros_like(v), trainable=False) for v in self._tracked_vars
        ]
        # Ω_k — consolidated importance weights (accumulated over all tasks).
        self._big_omega: List[tf.Variable] = [
            tf.Variable(tf.zeros_like(v), trainable=False) for v in self._tracked_vars
        ]
        # Gradients at the previous step (used to compute g_k · Δθ_k).
        self._prev_grads: Optional[List[tf.Tensor]] = None

    def _get_tracked_vars(self) -> List[tf.Variable]:
        """Variables subject to SI regularisation."""
        vars_ = list(self.actor.common_variables)
        if self.regularize_critic:
            vars_ += self.critic1.common_variables + self.critic2.common_variables
        return vars_

    # ------------------------------------------------------------------
    # Override apply_update to also accumulate ω
    # ------------------------------------------------------------------

    def apply_update(self, actor_gradients, critic_gradients, alpha_gradient) -> None:
        """Apply gradients and accumulate SI surrogates for tracked vars."""
        super().apply_update(actor_gradients, critic_gradients, alpha_gradient)
        self._accumulate_small_omega(actor_gradients, critic_gradients)

    def _accumulate_small_omega(self, actor_grads, critic_grads) -> None:
        """ω_k += -g_k · (θ_k^new − θ_k^old_step), where old_step is the
        value just before this gradient update."""
        # Build a flat gradient list aligned with _tracked_vars.
        grads = list(actor_grads) if actor_grads else []
        if self.regularize_critic and critic_grads:
            grads += list(critic_grads)

        for v, g, theta_star in zip(self._tracked_vars, grads, self._theta_star):
            if g is None:
                continue
            delta = v - theta_star  # cumulative parameter change from task start
            self._small_omega[self._tracked_vars.index(v)].assign_add(-g * delta)

    # ------------------------------------------------------------------
    # SAC hooks
    # ------------------------------------------------------------------

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        """SI quadratic penalty, active from the second task onward."""
        loss = tf.zeros([])
        for v, v_old, omega in zip(self._tracked_vars, self._theta_old, self._big_omega):
            loss += tf.reduce_sum(omega * (v - v_old) ** 2)
        coef = tf.cond(seq_idx > 0, lambda: self.cl_reg_coef, lambda: 0.0)
        return loss * coef

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        if current_task_idx > 0:
            self._consolidate()

        # Snapshot θ^* at the start of the new task.
        for theta_star, v in zip(self._theta_star, self._tracked_vars):
            theta_star.assign(v)
        # Reset small-omega accumulators.
        for omega in self._small_omega:
            omega.assign(tf.zeros_like(omega))

    def _consolidate(self) -> None:
        """Compute Ω_k and save θ_k^* → θ_k^old at the end of a task."""
        for v, theta_star, small_omega, big_omega, theta_old in zip(
            self._tracked_vars,
            self._theta_star,
            self._small_omega,
            self._big_omega,
            self._theta_old,
        ):
            delta_sq = (v - theta_star) ** 2
            new_omega = small_omega / (delta_sq + self.si_epsilon)
            big_omega.assign_add(tf.maximum(new_omega, 0.0))
            theta_old.assign(v)
