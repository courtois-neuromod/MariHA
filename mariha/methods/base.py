"""Composition-based continual learning method base class.

A :class:`CLMethod` is attached to an agent via ``agent.cl_method = ...``
after the agent is constructed.  The agent then invokes the method's
hooks at the appropriate points in its training loop.

Lifecycle (in order, per task)
------------------------------
1. ``cl_method.on_task_start(agent, task_idx)``
   Called from :meth:`BaseAgent.on_task_start`.  By the time this fires,
   the agent has already finished any task-boundary resets (replay
   buffer, network re-init, optimizer reset).  SI uses this hook to
   snapshot ``θ*`` and zero its small-omega accumulator.

2. (training proceeds — many gradient steps fire)

3. ``cl_method.on_task_end(agent, task_idx)``
   Called from :meth:`BaseAgent.on_task_end` BEFORE any task-boundary
   resets.  At this point the agent's data structures (replay buffer,
   etc.) still hold the just-finished task's data — the right moment
   for regularizer-style methods to compute parameter importance,
   PackNet to prune, and AGEM/DER/ClonEx to fill episodic memory.

Per-step hooks (called from inside ``update_step`` / ``learn_on_batch``)
-----------------------------------------------------------------------
- ``compute_loss_penalty(agent, task_idx) -> tf.Tensor``
  Scalar tensor added inside the agent's gradient tape.  Used by
  L2/EWC/MAS/SI.

- ``adjust_gradients(agent, grads_by_group, ...) -> grads_by_group``
  Modifies gradients before the optimizer step.  Used by AGEM
  (gradient projection), PackNet (mask zeroing), and DER/ClonEx
  (additive distillation gradients).

- ``get_episodic_batch(agent, task_idx) -> Optional[batch]``
  Provides a replay batch from past tasks; the agent forwards it to
  ``adjust_gradients``.  Used by AGEM, DER, ClonEx.

- ``after_gradient_step(agent, grads_by_group, task_idx) -> None``
  Called after the optimizer applied the gradients.  SI uses this to
  accumulate its omega surrogate.  Runs inside the agent's
  ``@tf.function``-traced update path, so any operations must be
  TensorFlow-compatible (no eager-only calls like ``.numpy()``).

All hooks are no-ops by default; subclasses override only the ones
they need.
"""

from __future__ import annotations

import argparse
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import tensorflow as tf

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


class CLMethod(ABC):
    """Base class for composition-based continual learning methods.

    Attach an instance to an agent with ``agent.cl_method = cl_method``;
    the agent will then call the relevant hooks during training.
    """

    #: Short identifier used for log lines and CLI selection.
    name: str = "base"

    # ------------------------------------------------------------------
    # Task-boundary hooks
    # ------------------------------------------------------------------

    def on_task_start(self, agent: "BaseAgent", task_idx: int) -> None:
        """Called at the start of each task, after the agent has reset state."""

    def on_task_end(self, agent: "BaseAgent", task_idx: int) -> None:
        """Called at the end of each task, BEFORE the agent resets state.

        Importance estimators (EWC/MAS/L2/SI), PackNet's pruning,
        and DER/ClonEx episodic-memory filling all happen here while
        the buffer still holds the task data.
        """

    # ------------------------------------------------------------------
    # Per-update-step hooks
    # ------------------------------------------------------------------

    def compute_loss_penalty(
        self, agent: "BaseAgent", *, task_idx: int
    ) -> tf.Tensor:
        """Scalar regularization term added to actor/critic loss.

        Override in subclasses implementing parameter regularization
        (L2, EWC, MAS, SI).  The agent adds the returned tensor inside
        its ``GradientTape`` so the penalty flows through the standard
        gradient path for both actor and critic losses (the parameter
        groups are non-overlapping, so no double-counting).

        Default: scalar zero.
        """
        return tf.zeros([])

    def adjust_gradients(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Dict[str, List[Optional[tf.Tensor]]]:
        """Modify gradients before the optimizer step.

        AGEM projects them onto the feasible half-space; PackNet masks
        out frozen weights; DER/ClonEx add a distillation gradient
        term.  ``grads_by_group`` keys mirror
        :meth:`BaseAgent.get_named_parameter_groups`.

        Default: identity.
        """
        return grads_by_group

    def get_episodic_batch(
        self, agent: "BaseAgent", *, task_idx: int
    ) -> Optional[Dict[str, tf.Tensor]]:
        """Provide a replay batch from past tasks for the current update.

        Forwarded by the agent to :meth:`adjust_gradients`.  Used by
        AGEM, DER, ClonEx.

        Default: ``None``.
        """
        return None

    def after_gradient_step(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        metrics: Optional[Dict[str, tf.Tensor]] = None,
    ) -> None:
        """Called after the optimizer has applied the adjusted gradients.

        Runs inside the agent's ``@tf.function``-traced update path,
        so any TF operations are baked into the graph.  SI uses this
        hook to accumulate its small-omega surrogate.

        ``metrics`` is the dict the agent's gradient step returns
        (losses, abs_error, q-values, ...).  It is ``None`` only if a
        caller invokes the hook outside the normal training step.
        Hooks that do not need it should accept and ignore the kwarg.

        Default: no-op.  Note: the explicit ``return`` statement
        below is required so AutoGraph can convert the method body
        when it gets called from inside an agent's
        ``@tf.function``-traced update path.  A docstring-only body
        confuses the converter into emitting an empty ``with``
        block.
        """
        return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, directory: Path) -> None:
        """Persist any internal state to ``directory``.  Default: no-op."""

    def load_state(self, directory: Path) -> None:
        """Restore state previously written by :meth:`save_state`.  Default: no-op."""

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CL-method-specific CLI flags to ``parser``.  Default: no-op."""

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        agent: "BaseAgent",
    ) -> "CLMethod":
        """Construct a CL method from parsed CLI args.

        Subclasses must implement this.  The returned instance is
        attached to ``agent`` by the caller (typically the runner
        script after agent construction).
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_args()."
        )
