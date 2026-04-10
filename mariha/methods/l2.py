"""L2 parameter regularization continual learning baseline.

Penalizes deviation from the previous task's parameters with uniform
importance weights — every parameter is treated as equally important.
The result is the simplest member of the EWC family: ``λ · Σ (θ − θ*)²``.

Reference: Kirkpatrick et al., 2017 (baseline variant without Fisher).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import tensorflow as tf

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.regularizer_base import ParameterRegularizer


@register_cl("l2")
class L2Regularizer(ParameterRegularizer):
    """Uniform-importance L2 anchor.

    Sets every importance weight to 1.0, so the loss penalty becomes
    ``λ · Σ (θ − θ*)²`` — the standard L2 baseline used as a reference
    point in the EWC paper.  Inherits all the bookkeeping from
    :class:`ParameterRegularizer`; the only behaviour overridden is
    importance computation, which is data-free for this method.

    Args mirror :class:`ParameterRegularizer`.
    """

    name = "l2"
    _needs_data = False

    def _compute_importance(
        self,
        agent,
        batch: Optional[Dict[str, tf.Tensor]],
    ) -> Dict[str, List[tf.Tensor]]:
        """Return ``ones_like`` for every parameter in every regularized group.

        Note that we read the *parameter shapes* from
        ``_reg_weights`` rather than from a fresh
        ``get_named_parameter_groups()`` call so that, even if a future
        agent rebuilds its variables between tasks, this method still
        produces correctly shaped tensors aligned with the storage the
        :meth:`ParameterRegularizer.compute_loss_penalty` reads from.
        """
        return {
            gn: [tf.ones_like(w) for w in self._reg_weights[gn]]
            for gn in self._regularize_groups
        }
