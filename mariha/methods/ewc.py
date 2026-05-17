"""Elastic Weight Consolidation (EWC) continual learning baseline.

Estimates parameter importance via the diagonal of the Fisher
information matrix, computed from the Jacobian of the network's
per-group outputs (actor logits, Q-values, value head, ...).  Importance
is averaged over a small batch of replay samples drawn from the
just-finished task; the resulting per-parameter weights anchor the
quadratic penalty added to the next task's loss.

Reference: Kirkpatrick et al., 2017 — *Overcoming catastrophic
forgetting in neural networks*, arXiv:1612.00796.

Agent-agnostic
--------------
Works on any agent that implements
:meth:`BaseAgent.get_named_parameter_groups` *and*
:meth:`BaseAgent.forward_for_importance`.  EWC was originally introduced
on DQN, where it reduces naturally to a single ``"q"`` group; the same
class also works for SAC's actor and PPO's policy head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import tensorflow as tf

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.regularizer_base import ParameterRegularizer

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("ewc")
class EWC(ParameterRegularizer):
    """Elastic Weight Consolidation.

    Per-parameter importance is the diagonal of the empirical Fisher
    information matrix:

    .. math::

        F_k = \\mathbb{E}_x \\left[ \\left( \\frac{\\partial \\log p(y\\,|\\,x)}{\\partial \\theta_k} \\right)^2 \\right]

    Approximated here by averaging the squared per-output Jacobian
    components — i.e. the same construction the original SAC
    implementation used.  This avoids materialising the full output
    distribution and matches the form documented in the original
    paper.

    Args mirror :class:`ParameterRegularizer`.
    """

    name = "ewc"

    def _compute_importance(
        self,
        agent: "BaseAgent",
        batch: Optional[Dict[str, tf.Tensor]],
    ) -> Dict[str, List[tf.Tensor]]:
        """Average squared Jacobian per parameter, clipped from below.

        For each regularized group ``g``, computes
        ``mean_x [ (∂ Σ_o output_g[o] / ∂ θ_k)² ]`` over the batch.
        The per-sample squared gradient of the summed output is the
        diagonal Fisher approximation; the outer mean averages over
        the replay batch.  Values below ``1e-5`` are clipped to avoid
        producing zero anchors for rarely-activated parameters (per
        the original implementation).
        """
        groups = agent.get_named_parameter_groups()
        with tf.GradientTape(persistent=True) as tape:
            outputs = agent.forward_for_importance(
                batch["obs"], batch["one_hot"]
            )
            # Sum each group's output along the per-action axis to get
            # one scalar-per-sample, then sum over the batch so the
            # outer ``tape.jacobian`` returns a (batch,)-prefixed
            # tensor.  Equivalent to the legacy ``g.jacobian(logits,
            # ...)`` flow.
            summed = {
                gn: tf.reduce_sum(outputs[gn], axis=-1)
                for gn in self._regularize_groups
            }

        importance: Dict[str, List[tf.Tensor]] = {}
        for gn in self._regularize_groups:
            gs = tape.jacobian(summed[gn], groups[gn], experimental_use_pfor=False)
            per_param: List[tf.Tensor] = []
            for var, g in zip(groups[gn], gs):
                if g is None:
                    # Defensive: a parameter that doesn't influence the
                    # group's output gets zero importance instead of a
                    # ``None`` that would crash the merge step below.
                    per_param.append(tf.zeros_like(var))
                    continue
                # ``g`` has shape (batch, *param_shape).  Square,
                # average over the batch axis, and clip from below.
                fisher = tf.reduce_mean(g ** 2, axis=0)
                fisher = tf.clip_by_value(fisher, 1e-5, np.inf)
                per_param.append(fisher)
            importance[gn] = per_param
        del tape
        return importance
