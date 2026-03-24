"""A-GEM (Averaged Gradient Episodic Memory) continual learning baseline.

Stores a small episodic memory of transitions from each past task.  At each
gradient step the current gradient is projected onto the half-space defined
by the reference gradient (computed from the episodic memory), ensuring
the update does not increase the loss on past tasks.

Reference: Chaudhry et al., 2019 — arXiv:1812.00420.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from mariha.replay.buffers import EpisodicMemory
from mariha.rl.sac import SAC


class AGEM_SAC(SAC):
    """A-GEM gradient projection method.

    Args:
        episodic_mem_per_task: Transitions to store per task in the episodic
            memory.  Total capacity = ``episodic_mem_per_task × num_tasks``.
        episodic_batch_size: Mini-batch size sampled from episodic memory to
            compute the reference gradient at each update step.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size

        episodic_mem_size = episodic_mem_per_task * self.num_tasks
        self.episodic_memory = EpisodicMemory(
            obs_shape=self.obs_shape,
            act_dim=self.act_dim,
            size=episodic_mem_size,
            num_tasks=self.num_tasks,
        )

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        if current_task_idx > 0 and self.replay_buffer.size >= self.episodic_mem_per_task:
            new_mem = self.replay_buffer.sample_batch(self.episodic_mem_per_task)
            # EpisodicMemory.store_multiple expects numpy arrays.
            new_mem_np = {k: v.numpy() for k, v in new_mem.items()}
            self.episodic_memory.store_multiple(**new_mem_np)

    def get_episodic_batch(
        self, current_task_idx: int
    ) -> Optional[Dict[str, tf.Tensor]]:
        if current_task_idx > 0 and self.episodic_memory.size > 0:
            return self.episodic_memory.sample_batch(self.episodic_batch_size)
        return None

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
            (ref_actor_grads, ref_critic_grads, _), _ = self.get_gradients(
                seq_idx=tf.constant(-1), **episodic_batch
            )
            dot_prod = sum(
                tf.reduce_sum(g * r)
                for g, r in zip(actor_gradients, ref_actor_grads)
                if g is not None and r is not None
            ) + sum(
                tf.reduce_sum(g * r)
                for g, r in zip(critic_gradients, ref_critic_grads)
                if g is not None and r is not None
            )
            ref_sq_norm = sum(
                tf.reduce_sum(r ** 2)
                for r in ref_actor_grads + ref_critic_grads
                if r is not None
            )
            violation = tf.cond(dot_prod >= 0, lambda: 0, lambda: 1)
            actor_gradients = self._project(actor_gradients, ref_actor_grads, dot_prod, ref_sq_norm)
            critic_gradients = self._project(critic_gradients, ref_critic_grads, dot_prod, ref_sq_norm)
            metrics["agem_violation"] = violation

        return actor_gradients, critic_gradients, alpha_gradient

    def _project(
        self,
        new_grads: List[tf.Tensor],
        ref_grads: List[tf.Tensor],
        dot_prod: tf.Tensor,
        ref_sq_norm: tf.Tensor,
    ) -> List[tf.Tensor]:
        """Project new_grads onto the feasible half-space."""
        return [
            tf.cond(
                dot_prod >= 0,
                lambda: g,
                lambda: g - (dot_prod / ref_sq_norm) * r,
            )
            for g, r in zip(new_grads, ref_grads)
            if g is not None
        ]
