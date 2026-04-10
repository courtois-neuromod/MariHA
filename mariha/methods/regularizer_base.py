"""Shared base for parameter-regularization continual learning methods.

:class:`ParameterRegularizer` factors out the bookkeeping that L2, EWC,
MAS, and SI all share so that each concrete method only has to implement
the importance computation.

The shared mechanics are:

1. **θ\\*  snapshots** — at every task boundary the current values of the
   regularized parameters are copied into ``_old_params`` (a dict of
   ``tf.Variable``).
2. **Importance weights** — running per-parameter scalars in
   ``_reg_weights``, accumulated across tasks via the same dict-of-
   ``tf.Variable`` storage.
3. **Quadratic penalty** — ``λ · Σ_g Σ_k w[g][k] · (θ[g][k] − θ*[g][k])²``,
   added inside the agent's ``GradientTape`` via
   :meth:`compute_loss_penalty`.

Lazy initialization
-------------------
The ``_old_params`` and ``_reg_weights`` dicts are populated the first
time :meth:`compute_loss_penalty` or :meth:`on_task_end` is invoked.
Initialization queries ``agent.get_named_parameter_groups()`` to learn
the variable shapes and names, then allocates one ``tf.Variable`` per
parameter.  This must happen exactly once because the agent's
``learn_on_batch`` ``tf.function`` captures the resulting Variables by
identity — re-allocating them would invalidate the trace.

The lazy path keeps the API ergonomic: the user creates the regularizer
with no shape information, attaches it to an already-constructed agent,
and the regularizer figures out the rest on first use.

Subclass contract
-----------------
Subclasses override :meth:`_compute_importance` to return a dict mapping
group name → list of per-parameter importance tensors aligned with that
group's variables.  L2 returns ``ones_like``; EWC returns the Fisher
diagonal; MAS returns the absolute output sensitivity.

Subclasses can also flip ``_needs_data`` to ``False`` (L2) so that
importance is computed without sampling from the replay buffer.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import tensorflow as tf

from mariha.methods.base import CLMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


class ParameterRegularizer(CLMethod):
    """Base class for L2/EWC/MAS/SI-style parameter regularization.

    Args:
        cl_reg_coef: Quadratic penalty coefficient ``λ``.  The full
            term added to the loss is
            ``λ · Σ_g Σ_k w[g][k] · (θ[g][k] − θ*[g][k])²``.
        regularize_groups: Subset of group names from
            :meth:`BaseAgent.get_named_parameter_groups` to regularize.
            ``None`` (default) auto-selects: tries ``"actor"``,
            ``"policy"``, ``"q"`` in that order and uses the first one
            present on the agent.  This gives the canonical
            "regularize the actor/policy network" behaviour for SAC and
            PPO and "regularize the Q-network" for DQN.
        importance_batches: Number of replay batches to average over
            when computing importance weights at task end.  Ignored
            when :attr:`_needs_data` is ``False`` (L2).
        importance_batch_size: Mini-batch size for each importance
            computation pass.
    """

    #: Subclasses set this to ``False`` if their importance computation
    #: does not require sampled data from the replay buffer (L2 uses
    #: uniform weights).
    _needs_data: bool = True

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        regularize_groups: Optional[Sequence[str]] = None,
        importance_batches: int = 10,
        importance_batch_size: int = 256,
    ) -> None:
        self.cl_reg_coef = float(cl_reg_coef)
        self._regularize_groups: Optional[List[str]] = (
            list(regularize_groups) if regularize_groups else None
        )
        self.importance_batches = int(importance_batches)
        self.importance_batch_size = int(importance_batch_size)

        # Lazily allocated on first :meth:`_lazy_init` call.
        self._old_params: Optional[Dict[str, List[tf.Variable]]] = None
        self._reg_weights: Optional[Dict[str, List[tf.Variable]]] = None

    # ------------------------------------------------------------------
    # Group resolution + lazy init
    # ------------------------------------------------------------------

    @property
    def regularize_groups(self) -> List[str]:
        """The list of group names this regularizer touches.

        Resolved on first access via the ``actor → policy → q`` rule
        unless an explicit ``regularize_groups`` argument was passed at
        construction time.  After :meth:`_lazy_init` runs the value is
        frozen.
        """
        if self._regularize_groups is None:
            raise RuntimeError(
                "regularize_groups accessed before _lazy_init() — "
                "call _lazy_init(agent) first."
            )
        return self._regularize_groups

    def _resolve_regularize_groups(self, agent: "BaseAgent") -> List[str]:
        """Return the default group selection for ``agent``.

        Picks the first present of ``actor``, ``policy``, ``q``.  Falls
        back to *all* groups if none of those keys exist (a bare custom
        agent should usually pass ``regularize_groups`` explicitly).
        """
        groups = agent.get_named_parameter_groups()
        for default in ("actor", "policy", "q"):
            if default in groups:
                return [default]
        return list(groups.keys())

    def _lazy_init(self, agent: "BaseAgent") -> None:
        """Allocate ``_old_params`` and ``_reg_weights`` if not already done.

        Safe to call from inside a ``tf.function`` trace: the body runs
        in eager Python at trace time, allocating fresh ``tf.Variable``
        objects whose identity is then captured by the trace.  Calling
        again is a no-op (the second-time guard returns immediately).
        """
        if self._old_params is not None:
            return
        if self._regularize_groups is None:
            self._regularize_groups = self._resolve_regularize_groups(agent)

        groups = agent.get_named_parameter_groups()
        missing = [g for g in self._regularize_groups if g not in groups]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: regularize_groups {missing} not "
                f"present on {type(agent).__name__}.  Available groups: "
                f"{sorted(groups.keys())}"
            )

        self._old_params = {
            gn: [
                tf.Variable(
                    tf.identity(v),
                    trainable=False,
                    name=f"clreg_old_{gn}_{i}",
                )
                for i, v in enumerate(groups[gn])
            ]
            for gn in self._regularize_groups
        }
        self._reg_weights = {
            gn: [
                tf.Variable(
                    tf.zeros_like(v),
                    trainable=False,
                    name=f"clreg_w_{gn}_{i}",
                )
                for i, v in enumerate(groups[gn])
            ]
            for gn in self._regularize_groups
        }

    # ------------------------------------------------------------------
    # CLMethod hooks
    # ------------------------------------------------------------------

    def compute_loss_penalty(
        self, agent: "BaseAgent", *, task_idx: int
    ) -> tf.Tensor:
        """Quadratic penalty added to the agent's loss.

        Returns ``tf.zeros([])`` on the first task — there is no
        ``θ*`` to anchor against yet.  Because ``task_idx`` is a Python
        integer (not a ``tf.Tensor``), this branch is evaluated at
        ``tf.function`` trace time and the zero is baked into the
        graph for the very first task's compiled update path.
        """
        if task_idx == 0:
            return tf.zeros([])
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        loss = tf.zeros([])
        for gn in self._regularize_groups:
            for v, v_old, w in zip(
                groups[gn], self._old_params[gn], self._reg_weights[gn]
            ):
                loss += tf.reduce_sum(w * (v - v_old) ** 2)
        return self.cl_reg_coef * loss

    def on_task_end(self, agent: "BaseAgent", task_idx: int) -> None:
        """Update importance weights and snapshot ``θ*`` for the next task.

        Fires *before* the runner resets the replay buffer (see the
        sequence diagram in :mod:`mariha.rl.base.training_loop`), so the
        importance computation has access to the just-finished task's
        data.
        """
        self._lazy_init(agent)
        new_weights = self._gather_importance_weights(agent)
        self._merge_weights(new_weights)
        # Snapshot current parameters as θ* for the next task.
        groups = agent.get_named_parameter_groups()
        for gn in self._regularize_groups:
            for old_v, cur_v in zip(self._old_params[gn], groups[gn]):
                old_v.assign(cur_v)

    # ------------------------------------------------------------------
    # Importance computation
    # ------------------------------------------------------------------

    def _gather_importance_weights(
        self, agent: "BaseAgent"
    ) -> Dict[str, List[tf.Tensor]]:
        """Return the new importance weights to merge in.

        For data-driven methods (EWC, MAS) this averages
        :meth:`_compute_importance` over ``importance_batches`` mini-
        batches drawn from the agent's replay buffer.  For data-free
        methods (L2) it calls :meth:`_compute_importance` once with
        ``batch=None``.
        """
        if not self._needs_data:
            return self._compute_importance(agent, batch=None)

        per_batch: List[Dict[str, List[tf.Tensor]]] = []
        for _ in range(self.importance_batches):
            batch = self._sample_importance_batch(agent)
            if batch is None:
                break
            per_batch.append(self._compute_importance(agent, batch=batch))

        if not per_batch:
            # Buffer was empty — return zeros (no contribution to merge).
            return {
                gn: [tf.zeros_like(v) for v in self._reg_weights[gn]]
                for gn in self._regularize_groups
            }

        return {
            gn: [
                tf.reduce_mean(tf.stack(weights, axis=0), axis=0)
                for weights in zip(*[b[gn] for b in per_batch])
            ]
            for gn in self._regularize_groups
        }

    def _sample_importance_batch(
        self, agent: "BaseAgent"
    ) -> Optional[Dict[str, tf.Tensor]]:
        """Sample one importance mini-batch from ``agent.replay_buffer``.

        Returns ``None`` if the agent has no replay buffer or its
        buffer is empty.  Subclasses can override to read from a
        different storage (per-scene pools, on-policy rollouts, etc.).
        """
        buf = getattr(agent, "replay_buffer", None)
        if buf is None or getattr(buf, "size", 0) == 0:
            return None
        return buf.sample_batch(min(self.importance_batch_size, buf.size))

    def _compute_importance(
        self,
        agent: "BaseAgent",
        batch: Optional[Dict[str, tf.Tensor]],
    ) -> Dict[str, List[tf.Tensor]]:
        """Subclass hook — return importance weights for one pass.

        Args:
            agent: The agent owning the parameter groups.  Subclasses
                typically call ``agent.forward_for_importance(obs,
                one_hot)`` to compute the per-group differentiable
                outputs that the Jacobian acts on.
            batch: One mini-batch from
                :meth:`_sample_importance_batch`, or ``None`` for
                data-free methods.

        Returns:
            Dict mapping group name to a list of importance tensors
            (one per parameter, same shape as the parameter).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _compute_importance()."
        )

    def _merge_weights(
        self, new_weights: Dict[str, List[tf.Tensor]]
    ) -> None:
        """Accumulate ``_reg_weights += new_weights`` (Kirkpatrick 2017)."""
        for gn in self._regularize_groups:
            for old_w, new_w in zip(self._reg_weights[gn], new_weights[gn]):
                old_w.assign_add(new_w)

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add the shared regularizer flags to ``parser``.

        Subclasses may add their own flags on top of these (e.g. SI's
        ``--si_epsilon``).  All flags are namespaced with the
        ``cl_*`` / ``regularize_*`` prefixes so they don't collide
        with the agent's own argument set.
        """
        parser.add_argument(
            "--cl_reg_coef",
            type=float,
            default=1.0,
            help="Quadratic penalty coefficient λ.",
        )
        parser.add_argument(
            "--regularize_groups",
            type=str,
            default=None,
            help=(
                "Comma-separated parameter-group names to regularize. "
                "Defaults to actor/policy/q (the first one present)."
            ),
        )
        parser.add_argument(
            "--importance_batches",
            type=int,
            default=10,
            help="Replay batches averaged for importance estimation.",
        )
        parser.add_argument(
            "--importance_batch_size",
            type=int,
            default=256,
            help="Mini-batch size for each importance estimation pass.",
        )

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        agent: "BaseAgent",
    ) -> "ParameterRegularizer":
        """Construct from a parsed argparse namespace."""
        groups = (
            [g.strip() for g in args.regularize_groups.split(",") if g.strip()]
            if getattr(args, "regularize_groups", None)
            else None
        )
        return cls(
            cl_reg_coef=getattr(args, "cl_reg_coef", 1.0),
            regularize_groups=groups,
            importance_batches=getattr(args, "importance_batches", 10),
            importance_batch_size=getattr(args, "importance_batch_size", 256),
        )
