"""Synaptic Intelligence (SI) continual learning baseline.

Unlike EWC and MAS — which compute importance *after* a task ends by
querying the replay buffer — SI accumulates importance *online* during
training.  Each gradient step contributes to a running surrogate
``ω_k += -g_k · Δθ_k`` that approximates the contribution of parameter
``θ_k`` to the loss decrease over the task.  At each task boundary the
surrogate is consolidated into a per-task importance weight:

.. math::

    \\Omega_k \\mathrel{+}= \\max\\!\\left(\\frac{\\omega_k}{(\\Delta\\theta_k^{\\text{total}})^2 + \\xi},\\ 0\\right)

and the reference parameters ``θ^*`` are saved for the quadratic
anchor on the next task.

Reference: Zenke, Poole & Ganguli, 2017 — *Continual Learning Through
Synaptic Intelligence*, arXiv:1703.04200.

Agent-agnostic
--------------
SI plugs into any agent that implements
:meth:`BaseAgent.get_named_parameter_groups`.  The per-step
accumulation runs from inside :meth:`after_gradient_step`, which fires
from inside the agent's compiled gradient step — so the ``tf.Variable``
state is allocated lazily on first use and captured by the trace.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import tensorflow as tf

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.base import CLMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("si")
class SI(CLMethod):
    """Synaptic Intelligence.

    Args:
        cl_reg_coef: Quadratic penalty coefficient ``λ``.  The full
            term added to the loss is
            ``λ · Σ_g Σ_k Ω[g][k] · (θ[g][k] − θ_old[g][k])²``.
        regularize_groups: Subset of group names from
            :meth:`BaseAgent.get_named_parameter_groups` to regularize.
            Defaults to the same ``actor → policy → q`` resolution as
            :class:`ParameterRegularizer`.
        si_epsilon: Damping constant ``ξ`` added to the squared
            parameter delta before dividing into ``ω``.  Prevents
            blow-ups for parameters that barely moved across a task.
    """

    name = "si"

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        regularize_groups: Optional[Sequence[str]] = None,
        si_epsilon: float = 0.1,
    ) -> None:
        self.cl_reg_coef = float(cl_reg_coef)
        self.si_epsilon = float(si_epsilon)
        self._regularize_groups: Optional[List[str]] = (
            list(regularize_groups) if regularize_groups else None
        )

        # Lazily allocated dicts of ``tf.Variable`` lists, populated on
        # first :meth:`_lazy_init` call.  Allocation must happen before
        # the first @tf.function trace so the Variables get captured
        # by identity.
        self._theta_star: Optional[Dict[str, List[tf.Variable]]] = None
        self._theta_old: Optional[Dict[str, List[tf.Variable]]] = None
        self._small_omega: Optional[Dict[str, List[tf.Variable]]] = None
        self._big_omega: Optional[Dict[str, List[tf.Variable]]] = None

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
        """Allocate ``θ*``, ``θ_old``, ``ω``, ``Ω`` dicts.

        Safe to call from inside a ``tf.function`` trace: the body runs
        in eager Python at trace time.  No-op on subsequent calls.
        """
        if self._theta_star is not None:
            return
        if self._regularize_groups is None:
            self._regularize_groups = self._resolve_regularize_groups(agent)

        groups = agent.get_named_parameter_groups()
        missing = [g for g in self._regularize_groups if g not in groups]
        if missing:
            raise ValueError(
                f"SI: regularize_groups {missing} not present on "
                f"{type(agent).__name__}.  Available groups: "
                f"{sorted(groups.keys())}"
            )

        def _alloc(prefix: str, init_value):
            return {
                gn: [
                    tf.Variable(
                        init_value(v),
                        trainable=False,
                        name=f"si_{prefix}_{gn}_{i}",
                    )
                    for i, v in enumerate(groups[gn])
                ]
                for gn in self._regularize_groups
            }

        self._theta_star = _alloc("theta_star", tf.identity)
        self._theta_old = _alloc("theta_old", tf.identity)
        self._small_omega = _alloc("small_omega", tf.zeros_like)
        self._big_omega = _alloc("big_omega", tf.zeros_like)

    # ------------------------------------------------------------------
    # CLMethod hooks
    # ------------------------------------------------------------------

    def compute_loss_penalty(
        self, agent: "BaseAgent", *, task_idx: int
    ) -> tf.Tensor:
        """``λ · Σ_g Σ_k Ω[g][k] · (θ[g][k] − θ_old[g][k])²``.

        Returns ``tf.zeros([])`` on the first task — there is no
        ``θ_old`` to anchor against yet.  ``task_idx`` is a Python int
        so this branch evaluates at trace time and the zero is baked
        into the first task's compiled graph.
        """
        if task_idx == 0:
            return tf.zeros([])
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        loss = tf.zeros([])
        for gn in self._regularize_groups:
            for v, v_old, omega in zip(
                groups[gn], self._theta_old[gn], self._big_omega[gn]
            ):
                loss += tf.reduce_sum(omega * (v - v_old) ** 2)
        return self.cl_reg_coef * loss

    def after_gradient_step(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        metrics: Optional[Dict[str, tf.Tensor]] = None,
    ) -> None:
        """Accumulate ``ω_k += -g_k · (θ_k − θ_k^*)``.

        Runs inside the agent's @tf.function-traced update path.  The
        delta is computed using the *post-update* parameter value (the
        agent has already applied the gradient by the time this hook
        fires) and the per-task reference snapshot ``θ*``.

        ``grads_by_group`` keys must match ``regularize_groups``;
        gradient lists for groups SI doesn't track are ignored.
        """
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        for gn in self._regularize_groups:
            if gn not in grads_by_group:
                continue
            for v, theta_star, omega, g in zip(
                groups[gn],
                self._theta_star[gn],
                self._small_omega[gn],
                grads_by_group[gn],
            ):
                if g is None:
                    continue
                omega.assign_add(-g * (v - theta_star))

    def on_task_end(self, agent: "BaseAgent", task_idx: int) -> None:
        """Consolidate ``Ω`` and snapshot the just-finished task's params.

        Fires from the runner *before* the agent's task-boundary
        reset, so the current parameter values are still those of the
        just-finished task.  After consolidation, ``θ_old`` is the
        anchor used by the next task's penalty.
        """
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        for gn in self._regularize_groups:
            for v, theta_star, small, big, theta_old in zip(
                groups[gn],
                self._theta_star[gn],
                self._small_omega[gn],
                self._big_omega[gn],
                self._theta_old[gn],
            ):
                delta_sq = (v - theta_star) ** 2
                new_omega = small / (delta_sq + self.si_epsilon)
                big.assign_add(tf.maximum(new_omega, 0.0))
                theta_old.assign(v)

    def on_task_start(self, agent: "BaseAgent", task_idx: int) -> None:
        """Snapshot the new task's starting parameters and reset ``ω``."""
        self._lazy_init(agent)
        groups = agent.get_named_parameter_groups()
        for gn in self._regularize_groups:
            for v, theta_star, small in zip(
                groups[gn],
                self._theta_star[gn],
                self._small_omega[gn],
            ):
                theta_star.assign(v)
                small.assign(tf.zeros_like(small))

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cl_reg_coef",
            type=float,
            default=1.0,
            help="SI quadratic penalty coefficient λ.",
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
            "--si_epsilon",
            type=float,
            default=0.1,
            help="Damping ξ added to (Δθ)² before dividing into ω.",
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, agent: "BaseAgent"
    ) -> "SI":
        groups = (
            [g.strip() for g in args.regularize_groups.split(",") if g.strip()]
            if getattr(args, "regularize_groups", None)
            else None
        )
        return cls(
            cl_reg_coef=getattr(args, "cl_reg_coef", 1.0),
            regularize_groups=groups,
            si_epsilon=getattr(args, "si_epsilon", 0.1),
        )
