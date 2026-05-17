"""Memory Aware Synapses (MAS) continual learning baseline.

Estimates parameter importance as the sensitivity of the network's
output magnitude to small parameter perturbations — computed as the
gradient of the squared L2 norm of the per-group outputs.

Reference: Aljundi et al., 2018 — *Memory Aware Synapses: Learning what
(not) to forget*, arXiv:1711.09601.

Agent-agnostic
--------------
Uses the same :meth:`BaseAgent.forward_for_importance` hook as
:class:`mariha.methods.ewc.EWC`, so any agent that supports EWC also
supports MAS — including DQN (single ``"q"`` group), SAC (actor +
critics), and PPO (single ``"policy"`` group).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import tensorflow as tf

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.regularizer_base import ParameterRegularizer

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("mas")
class MAS(ParameterRegularizer):
    """Memory Aware Synapses.

    Per-parameter importance is the average absolute Jacobian of the
    squared output norm:

    .. math::

        \\Omega_k = \\mathbb{E}_x \\left| \\frac{\\partial \\| f(x) \\|_2^2}{\\partial \\theta_k} \\right|

    Larger values mean perturbing that parameter would change the
    output more, so it deserves a stronger anchor on the next task.

    Args mirror :class:`ParameterRegularizer`.
    """

    name = "mas"

    def _compute_importance(
        self,
        agent: "BaseAgent",
        batch: Optional[Dict[str, tf.Tensor]],
    ) -> Dict[str, List[tf.Tensor]]:
        """Average absolute Jacobian of the squared output norm."""
        groups = agent.get_named_parameter_groups()
        with tf.GradientTape(persistent=True) as tape:
            outputs = agent.forward_for_importance(
                batch["obs"], batch["one_hot"]
            )
            # ‖output‖² summed along the per-action axis, giving one
            # scalar per sample.  ``tape.jacobian`` then returns a
            # (batch,)-prefixed tensor we average over.
            sq_norms = {
                gn: tf.reduce_sum(outputs[gn] ** 2, axis=-1)
                for gn in self._regularize_groups
            }

        importance: Dict[str, List[tf.Tensor]] = {}
        for gn in self._regularize_groups:
            gs = tape.jacobian(sq_norms[gn], groups[gn], experimental_use_pfor=False)
            per_param: List[tf.Tensor] = []
            for var, g in zip(groups[gn], gs):
                if g is None:
                    per_param.append(tf.zeros_like(var))
                    continue
                # ``g`` has shape (batch, *param_shape).  Take the
                # element-wise absolute value and average over the
                # batch axis — this is MAS's L1-style sensitivity
                # measure.
                per_param.append(tf.reduce_mean(tf.abs(g), axis=0))
            importance[gn] = per_param
        del tape
        return importance
