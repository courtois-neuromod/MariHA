"""PackNet continual learning baseline.

At the end of each task PackNet prunes a fixed fraction
(``prune_perc``) of the currently free actor weights — the lowest-
magnitude ones — and freezes them at their current values.  Future
tasks train only on the remaining unfrozen weights via three
mechanisms:

1. **Gradient masking** in :meth:`adjust_gradients` zeroes out the
   gradient of every frozen parameter so the optimizer cannot push
   them further.
2. **Adam-drift correction** in :meth:`after_gradient_step` re-asserts
   the frozen value after the optimizer step (Adam's momentum/RMS
   slot variables can drift even when the gradient is zero).
3. **Magnitude pruning** in :meth:`on_task_end` selects the lowest-
   magnitude *free* weights and adds them to the frozen set.

Unlike COOM's dynamic ``num_tasks_left / (num_tasks_left + 1)``
schedule, MariHA uses a fixed ``prune_perc`` because the benchmark
has 313+ scenes and the total task count is not known in advance.

Reference: Mallya & Lazebnik, 2018 — *PackNet: Adding Multiple Tasks
to a Single Network by Iterative Pruning*, arXiv:1711.05769.

Agent-agnostic
--------------
PackNet operates on whatever group :meth:`BaseAgent.get_named_parameter_groups`
exposes for the agent's actor — for SAC the ``"actor"`` group, for
PPO the ``"policy"`` group, and for DQN the ``"q"`` group (the
single trainable network).  Selection follows the same default
``actor → policy → q`` resolution as :class:`ParameterRegularizer`.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.base import CLMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("packnet")
class PackNet(CLMethod):
    """PackNet iterative magnitude pruning.

    Args:
        prune_perc: Fraction of currently *free* weights to prune at
            each task boundary.  Must be in ``(0, 1)``.  A value of
            0.5 halves the active capacity at every task switch.
        regularize_groups: Group names to prune.  Defaults to a
            single group from the ``actor → policy → q`` resolution.
    """

    name = "packnet"

    def __init__(
        self,
        prune_perc: float = 0.5,
        regularize_groups: Optional[Sequence[str]] = None,
    ) -> None:
        if not 0.0 < prune_perc < 1.0:
            raise ValueError(
                f"prune_perc must be in (0, 1), got {prune_perc}"
            )
        self.prune_perc = float(prune_perc)
        self._regularize_groups: Optional[List[str]] = (
            list(regularize_groups) if regularize_groups else None
        )

        # Lazily allocated dicts of ``tf.Variable`` lists, populated
        # on the first ``_lazy_init`` call.
        self._free_masks: Optional[Dict[str, List[tf.Variable]]] = None
        self._frozen_weights: Optional[Dict[str, List[tf.Variable]]] = None
        # Python flag (not a tf.Variable) — recompiled into the
        # per-task @tf.function trace via the closure capture in
        # the agent's _make_*_step_fn.
        self._has_frozen: bool = False

    # ------------------------------------------------------------------
    # Group resolution + lazy init
    # ------------------------------------------------------------------

    def _resolve_regularize_groups(self, agent: "BaseAgent") -> List[str]:
        groups = agent.get_named_parameter_groups()
        for default in ("actor", "policy", "q"):
            if default in groups:
                return [default]
        return list(groups.keys())

    def _lazy_init(self, agent: "BaseAgent") -> None:
        """Allocate ``_free_masks`` and ``_frozen_weights`` dicts.

        Safe to call from inside a ``tf.function`` trace: the body
        runs in eager Python at trace time so the resulting Variables
        are captured by identity.
        """
        if self._free_masks is not None:
            return
        if self._regularize_groups is None:
            self._regularize_groups = self._resolve_regularize_groups(agent)

        groups = agent.get_named_parameter_groups()
        missing = [g for g in self._regularize_groups if g not in groups]
        if missing:
            raise ValueError(
                f"PackNet: regularize_groups {missing} not present on "
                f"{type(agent).__name__}.  Available groups: "
                f"{sorted(groups.keys())}"
            )

        self._free_masks = {
            gn: [
                tf.Variable(
                    tf.ones_like(v),
                    trainable=False,
                    dtype=tf.float32,
                    name=f"packnet_mask_{gn}_{i}",
                )
                for i, v in enumerate(groups[gn])
            ]
            for gn in self._regularize_groups
        }
        self._frozen_weights = {
            gn: [
                tf.Variable(
                    tf.zeros_like(v),
                    trainable=False,
                    name=f"packnet_frozen_{gn}_{i}",
                )
                for i, v in enumerate(groups[gn])
            ]
            for gn in self._regularize_groups
        }

    # ------------------------------------------------------------------
    # CLMethod hooks
    # ------------------------------------------------------------------

    def adjust_gradients(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Dict[str, List[Optional[tf.Tensor]]]:
        """Zero out gradients of frozen parameters.

        On the first task no weights are frozen yet, so the masks are
        all ones and this is a no-op.  On subsequent tasks the masks
        carry the cumulative free-weight pattern.
        """
        self._lazy_init(agent)
        out: Dict[str, List[Optional[tf.Tensor]]] = dict(grads_by_group)
        for gn in self._regularize_groups:
            if gn not in grads_by_group:
                continue
            masked: List[Optional[tf.Tensor]] = []
            for g, m in zip(grads_by_group[gn], self._free_masks[gn]):
                if g is None:
                    masked.append(None)
                else:
                    masked.append(g * m)
            out[gn] = masked
        return out

    def after_gradient_step(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        metrics: Optional[Dict[str, tf.Tensor]] = None,
    ) -> None:
        """Restore frozen weights after the optimizer step.

        Even with zero gradient Adam can drift a parameter via its
        first/second-moment slot variables.  We re-assert the frozen
        value as ``var = var * mask + frozen``, which is a no-op for
        free weights (mask=1, frozen=0) and snaps frozen weights back
        to their stored value (mask=0, frozen=stored).

        Skipped at trace time when no weights have been frozen yet
        (the ``_has_frozen`` Python flag is captured by the per-task
        recompile, so the first task's compiled graph contains no
        restoration code).
        """
        if not self._has_frozen:
            return
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        for gn in self._regularize_groups:
            for v, mask, frozen in zip(
                groups[gn],
                self._free_masks[gn],
                self._frozen_weights[gn],
            ):
                v.assign(v * mask + frozen)

    def on_task_end(self, agent: "BaseAgent", task_idx: int) -> None:
        """Prune ``prune_perc`` of currently free weights by lowest magnitude.

        Each variable is pruned independently — we never compare
        magnitudes across layers, matching the original PackNet
        formulation.  After this hook the agent's ``on_task_change``
        will recompile its ``_make_*_step_fn`` for the next task,
        and the new trace picks up the updated ``_has_frozen`` flag.
        """
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        for gn in self._regularize_groups:
            for var, mask, frozen in zip(
                groups[gn],
                self._free_masks[gn],
                self._frozen_weights[gn],
            ):
                var_np = var.numpy()
                mask_np = mask.numpy()

                free_indices = np.where(mask_np.flatten() > 0.5)[0]
                if len(free_indices) == 0:
                    continue

                n_prune = max(1, int(len(free_indices) * self.prune_perc))
                magnitudes = np.abs(var_np.flatten())
                free_mags = magnitudes[free_indices]

                # Pick the n_prune lowest-magnitude free weights.
                prune_local_idx = np.argpartition(
                    free_mags, n_prune - 1
                )[:n_prune]
                prune_global_idx = free_indices[prune_local_idx]

                flat_mask = mask_np.flatten().copy()
                flat_mask[prune_global_idx] = 0.0
                new_mask = flat_mask.reshape(var_np.shape)
                mask.assign(new_mask)
                # Frozen tensor stores the value at pruning time —
                # masked out for currently-free weights so the
                # restoration formula ``var * mask + frozen`` works.
                frozen.assign(var_np * (1.0 - new_mask))

        self._has_frozen = True
        agent.logger.log(
            f"[packnet] Pruned {self.prune_perc * 100:.0f}% of free "
            f"weights in groups {self._regularize_groups}.",
            color="cyan",
        )

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--prune_perc",
            type=float,
            default=0.5,
            help="Fraction of free weights to prune at each task boundary.",
        )
        parser.add_argument(
            "--regularize_groups",
            type=str,
            default=None,
            help=(
                "Comma-separated parameter-group names to prune. "
                "Defaults to actor/policy/q (the first one present)."
            ),
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, agent: "BaseAgent"
    ) -> "PackNet":
        groups = (
            [g.strip() for g in args.regularize_groups.split(",") if g.strip()]
            if getattr(args, "regularize_groups", None)
            else None
        )
        return cls(
            prune_perc=getattr(args, "prune_perc", 0.5),
            regularize_groups=groups,
        )
