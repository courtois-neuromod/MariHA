"""ClonEx-SAC (Clone-and-Explore) continual learning baseline.

Extends DER++ by distilling both the actor policy *and* the critic Q-values
from episodic memory.  The actor distillation uses KL divergence (same as
DER++); the critic distillation uses MSE between the stored and current
Q-value predictions.

Only the buffer-based variant is supported here (``episodic_memory_from_buffer
= True``), which re-uses transitions already in the replay buffer rather than
requiring access to past environments.

Reference: Ben-Iwhiwhu et al., 2021 — arXiv:2105.07748.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from mariha.replay.buffers import EpisodicMemory
from mariha.rl.sac import SAC


class ClonEx_SAC(SAC):
    """ClonEx: actor + critic distillation from episodic memory.

    Args:
        episodic_mem_per_task: Transitions stored per completed task.
        episodic_batch_size: Mini-batch sampled from episodic memory each step.
        der_alpha: Weight of the KL actor-distillation loss.
        clone_critic: If ``True`` (default), also distil critic Q-values.
        critic_alpha: Weight of the MSE critic-distillation loss.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        der_alpha: float = 0.1,
        clone_critic: bool = True,
        critic_alpha: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size
        self.der_alpha = der_alpha
        self.clone_critic = clone_critic
        self.critic_alpha = critic_alpha

        episodic_mem_size = episodic_mem_per_task * self.num_tasks
        self.episodic_memory = EpisodicMemory(
            obs_shape=self.obs_shape,
            act_dim=self.act_dim,
            size=episodic_mem_size,
            num_tasks=self.num_tasks,
            save_targets=True,
        )

    # ------------------------------------------------------------------
    # SAC lifecycle hooks
    # ------------------------------------------------------------------

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        if current_task_idx > 0 and self.replay_buffer.size >= self.episodic_mem_per_task:
            self._fill_episodic_memory()

    def get_episodic_batch(
        self, current_task_idx: int
    ) -> Optional[Dict[str, tf.Tensor]]:
        if current_task_idx > 0 and self.episodic_memory.size > 0:
            return self.episodic_memory.sample_batch(self.episodic_batch_size)
        return None

    # ------------------------------------------------------------------
    # Gradient adjustment
    # ------------------------------------------------------------------

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient,
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Tuple:
        if current_task_idx > 0 and episodic_batch is not None:
            kl_grads, kl_loss = self._kl_actor_gradients(episodic_batch)
            actor_gradients = [
                g + k if g is not None and k is not None else g
                for g, k in zip(actor_gradients, kl_grads)
            ]
            metrics["kl_loss"] = kl_loss

            if self.clone_critic:
                c_grads, c_loss = self._mse_critic_gradients(episodic_batch)
                critic_gradients = [
                    g + c if g is not None and c is not None else g
                    for g, c in zip(critic_gradients, c_grads)
                ]
                metrics["reg_loss"] = metrics.get("reg_loss", tf.zeros([])) + c_loss

        return actor_gradients, critic_gradients, alpha_gradient

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_episodic_memory(self) -> None:
        """Sample from replay buffer and store with current network predictions."""
        batch = self.replay_buffer.sample_batch(self.episodic_mem_per_task)
        obs = batch["obs"]
        one_hot = batch["one_hot"]

        actor_logits = self.actor(obs, one_hot).numpy()
        q1_preds = self.critic1(obs, one_hot).numpy()
        q2_preds = self.critic2(obs, one_hot).numpy()

        self.episodic_memory.store_multiple(
            obs=obs.numpy(),
            actions=batch["actions"].numpy(),
            rewards=batch["rewards"].numpy(),
            next_obs=batch["next_obs"].numpy(),
            done=batch["done"].numpy(),
            one_hot=one_hot.numpy(),
            actor_logits=actor_logits,
            critic1_preds=q1_preds,
            critic2_preds=q2_preds,
        )

    def _kl_actor_gradients(
        self, episodic_batch: Dict[str, tf.Tensor]
    ) -> Tuple[List[Optional[tf.Tensor]], tf.Tensor]:
        """KL(stored_logits ‖ current_logits) — same as DER++."""
        with tf.GradientTape() as tape:
            current_logits = self.actor(
                episodic_batch["obs"], episodic_batch["one_hot"]
            )
            stored_logits = episodic_batch["actor_logits"]
            p = tf.nn.softmax(stored_logits)
            kl = tf.reduce_mean(
                tf.reduce_sum(
                    p * (tf.nn.log_softmax(stored_logits) - tf.nn.log_softmax(current_logits)),
                    axis=-1,
                )
            )
            kl_loss = self.der_alpha * kl
        grads = tape.gradient(kl_loss, self.actor.trainable_variables)
        return grads, kl_loss

    def _mse_critic_gradients(
        self, episodic_batch: Dict[str, tf.Tensor]
    ) -> Tuple[List[Optional[tf.Tensor]], tf.Tensor]:
        """MSE(stored_Q ‖ current_Q) for both critics."""
        obs = episodic_batch["obs"]
        one_hot = episodic_batch["one_hot"]
        q1_target = episodic_batch["critic1_preds"]
        q2_target = episodic_batch["critic2_preds"]

        with tf.GradientTape() as tape:
            q1_current = self.critic1(obs, one_hot)
            q2_current = self.critic2(obs, one_hot)
            mse = (
                tf.reduce_mean((q1_current - tf.stop_gradient(q1_target)) ** 2)
                + tf.reduce_mean((q2_current - tf.stop_gradient(q2_target)) ** 2)
            )
            mse_loss = self.critic_alpha * mse

        critic_vars = self.critic1.trainable_variables + self.critic2.trainable_variables
        grads = tape.gradient(mse_loss, critic_vars)
        return grads, mse_loss
