"""Dark Experience Replay++ (DER++) continual learning baseline.

At each task boundary a snapshot of the replay buffer is stored in an
episodic memory together with the actor's logits at that time.  During
subsequent tasks, a KL-divergence distillation loss anchors the current
policy to the stored logits, preventing the actor from forgetting past
behaviour.

The distillation gradient is added directly to the actor gradients inside
``adjust_gradients``, keeping the mechanism transparent and easy to compose
with other methods.

Reference: Buzzega et al., 2020 — arXiv:2004.07211.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from mariha.replay.buffers import EpisodicMemory
from mariha.rl.sac import SAC


class DER_SAC(SAC):
    """DER++ behavioural-cloning distillation method.

    Args:
        episodic_mem_per_task: Transitions stored per completed task.
        episodic_batch_size: Mini-batch size sampled from episodic memory
            at each update step.
        der_alpha: Weight of the KL distillation loss.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        der_alpha: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size
        self.der_alpha = der_alpha

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
    # Gradient adjustment: add KL distillation term to actor gradients
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
        return actor_gradients, critic_gradients, alpha_gradient

    # ------------------------------------------------------------------
    # Internal helpers (numpy — called outside @tf.function)
    # ------------------------------------------------------------------

    def _fill_episodic_memory(self) -> None:
        """Sample from replay buffer and store with current actor logits."""
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
        """Compute KL(stored_logits ‖ current_logits) gradients w.r.t. actor.

        KL(p ‖ q) = Σ p · (log p − log q), where p = softmax(stored),
        q = softmax(current).
        """
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
