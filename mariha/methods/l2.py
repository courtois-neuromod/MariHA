"""L2 regularisation continual learning baseline.

Penalises deviation from the previous task's parameters with uniform
importance weights (i.e., all weights matter equally).

Reference: Kirkpatrick et al., 2017 (baseline variant without Fisher).
"""

from __future__ import annotations

from typing import List

import tensorflow as tf

from mariha.methods.regularization import Regularization_SAC
from mariha.replay.buffers import ReplayBuffer


class L2_SAC(Regularization_SAC):
    """L2 regularisation method.

    Uniform importance weights — all parameters penalised equally.

    Args:
        cl_reg_coef: Regularisation coefficient λ.
        regularize_critic: Also regularise critic weights.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        if self.regularize_critic:
            return [tf.ones_like(p) for p in self.all_common_variables]
        return (
            [tf.ones_like(p) for p in self.actor_common_variables]
            + [tf.zeros_like(p) for p in self.critic_common_variables]
        )

    def _update_reg_weights(
        self,
        replay_buffer: ReplayBuffer,
        batches_num: int = 10,
        batch_size: int = 256,
    ) -> None:
        """L2 does not need data — just set weights to 1 (or 0 for critic)."""
        self._merge_weights(self._get_importance_weights())
