"""PackNet continual learning baseline.

At the end of each task a fixed fraction (``prune_perc``) of the
currently active actor weights are pruned — frozen at their current values
and protected from future gradient updates.  New tasks train only on the
remaining free weights.

Unlike COOM's dynamic ``num_tasks_left / (num_tasks_left + 1)`` schedule,
MariHA uses a fixed ``prune_perc`` because the benchmark has 313+ scenes
and the total task count is not known in advance for scheduling.

Reference: Mallya & Lazebnik, 2018 — arXiv:1711.05769.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from mariha.rl.sac import SAC


class PackNet_SAC(SAC):
    """PackNet iterative pruning for the actor network.

    Args:
        prune_perc: Fraction of currently free actor weights to prune at
            each task boundary.  Must be in (0, 1).
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        prune_perc: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 < prune_perc < 1:
            raise ValueError(f"prune_perc must be in (0, 1), got {prune_perc}")
        self.prune_perc = prune_perc

        # 1.0 = free to update, 0.0 = frozen.
        self._free_masks: List[tf.Variable] = [
            tf.Variable(tf.ones_like(v), trainable=False, dtype=tf.float32)
            for v in self.actor.trainable_variables
        ]
        # Stores the frozen parameter values so Adam drift can be corrected.
        self._frozen_weights: List[tf.Variable] = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.actor.trainable_variables
        ]
        # Whether any weights have been pruned yet (tracked at Python level
        # so @tf.function can specialise — recompiled at each task change).
        self._has_frozen = False

    # ------------------------------------------------------------------
    # Override apply_update to restore frozen weights after Adam step
    # ------------------------------------------------------------------

    def apply_update(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: Optional[tf.Tensor],
    ) -> None:
        """Apply gradients then restore Adam-drifted frozen weights."""
        super().apply_update(actor_gradients, critic_gradients, alpha_gradient)
        if self._has_frozen:
            for var, mask, frozen in zip(
                self.actor.trainable_variables, self._free_masks, self._frozen_weights
            ):
                # var = var * free_mask + frozen * pruned_mask
                var.assign(var * mask + frozen)

    # ------------------------------------------------------------------
    # CL extension points
    # ------------------------------------------------------------------

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient,
        current_task_idx: int,
        metrics: dict,
        episodic_batch=None,
    ) -> Tuple:
        """Zero out gradients for frozen (pruned) weights."""
        masked = [
            g * m if g is not None else g
            for g, m in zip(actor_gradients, self._free_masks)
        ]
        return masked, critic_gradients, alpha_gradient

    def on_task_end(self, current_task_idx: int) -> None:
        super().on_task_end(current_task_idx)
        self._prune_actor()

    # ------------------------------------------------------------------
    # Pruning (numpy — called outside @tf.function)
    # ------------------------------------------------------------------

    def _prune_actor(self) -> None:
        """Prune ``prune_perc`` of free actor weights by lowest magnitude."""
        for i, (var, mask, frozen) in enumerate(
            zip(self.actor.trainable_variables, self._free_masks, self._frozen_weights)
        ):
            var_np = var.numpy()
            mask_np = mask.numpy()

            free_indices = np.where(mask_np.flatten() > 0.5)[0]
            if len(free_indices) == 0:
                continue

            n_prune = max(1, int(len(free_indices) * self.prune_perc))
            magnitudes = np.abs(var_np.flatten())
            free_mags = magnitudes[free_indices]

            # Select the n_prune lowest-magnitude free weights.
            prune_local_idx = np.argpartition(free_mags, n_prune - 1)[:n_prune]
            prune_global_idx = free_indices[prune_local_idx]

            flat_mask = mask_np.flatten().copy()
            flat_mask[prune_global_idx] = 0.0
            new_mask = flat_mask.reshape(var_np.shape)
            mask.assign(new_mask)

            # Record frozen values — pruned weights stay at current values.
            frozen.assign(var_np * (1.0 - new_mask))

        self._has_frozen = True
        self.logger.log(
            f"PackNet: pruned {self.prune_perc*100:.0f}% of free actor weights.",
            color="cyan",
        )
