"""A-GEM (Averaged Gradient Episodic Memory) continual learning baseline.

Stores a small episodic memory of transitions sampled at every task
boundary.  At each gradient step the current gradient is projected
onto the half-space defined by a *reference gradient* computed from
the episodic memory — the projection guarantees the update does not
*increase* the loss on past tasks (Chaudhry et al. 2019).

For groups ``g`` where ``⟨g_new, g_ref⟩ < 0`` the projected gradient is

.. math::

    \\tilde g_{\\text{new}} = g_{\\text{new}}
        - \\frac{\\langle g_{\\text{new}}, g_{\\text{ref}} \\rangle}{\\| g_{\\text{ref}} \\|^2}\\, g_{\\text{ref}}

otherwise it is left untouched.

Reference: Chaudhry et al., 2019 — *Efficient Lifelong Learning with
A-GEM*, arXiv:1812.00420.

Agent-agnostic
--------------
AGEM works on any agent that implements
:meth:`BaseAgent._compute_reference_loss`.  SAC, DQN, and PPO all
support it: SAC and DQN reuse their standard TD/actor losses on raw
transitions, while PPO falls back to a behaviour-cloning loss
(``-mean log π(a|s)``) on stored actions — the same on-policy
fallback used by Rolnick et al. 2019.

Episodic-memory format
----------------------
The memory stores raw transitions ``(obs, next_obs, actions, rewards,
done, one_hot)``.  This is the universal RL transition format and
matches the dict shape that SAC's and DQN's replay buffers produce.
PPO ignores the ``next_obs/rewards/done`` fields.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Dict, List, Optional

import tensorflow as tf

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.base import CLMethod
from mariha.replay.buffers import EpisodicMemory

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("agem")
class AGEM(CLMethod):
    """Averaged Gradient Episodic Memory.

    Args:
        episodic_mem_per_task: Transitions stored per task in the
            episodic memory.  Total capacity = ``episodic_mem_per_task
            × num_tasks``.
        episodic_batch_size: Mini-batch size sampled from episodic
            memory at each gradient step to compute the reference
            gradient.
        regularize_groups: Subset of group names from
            :meth:`BaseAgent.get_named_parameter_groups` to project.
            Defaults to *all* groups in
            :meth:`BaseAgent._compute_reference_loss`'s output (i.e.,
            actor + critics for SAC, the single ``"q"`` group for DQN,
            the ``"policy"`` group for PPO).
    """

    name = "agem"

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        regularize_groups: Optional[List[str]] = None,
    ) -> None:
        self.episodic_mem_per_task = int(episodic_mem_per_task)
        self.episodic_batch_size = int(episodic_batch_size)
        self._regularize_groups: Optional[List[str]] = (
            list(regularize_groups) if regularize_groups else None
        )
        self._episodic_memory: Optional[EpisodicMemory] = None

    # ------------------------------------------------------------------
    # Lazy init + group resolution
    # ------------------------------------------------------------------

    def _lazy_init(self, agent: "BaseAgent") -> None:
        """Allocate ``EpisodicMemory`` and resolve regularize groups.

        Sized to ``episodic_mem_per_task × num_tasks`` so the buffer
        never overwrites old entries — AGEM relies on uniform task
        coverage in the reference batch.
        """
        if self._episodic_memory is not None:
            return
        self._episodic_memory = EpisodicMemory(
            obs_shape=agent.obs_shape,
            act_dim=agent.act_dim,
            size=self.episodic_mem_per_task * agent.num_tasks,
            num_tasks=agent.num_tasks,
        )
        if self._regularize_groups is None:
            # Default: project all groups the agent's reference loss
            # touches.  Probe with a tiny synthetic batch — we just
            # need the keys, not the actual loss values.
            self._regularize_groups = list(
                agent.get_named_parameter_groups().keys()
            )

    # ------------------------------------------------------------------
    # CLMethod hooks
    # ------------------------------------------------------------------

    def on_task_start(self, agent: "BaseAgent", task_idx: int) -> None:
        """Refill episodic memory with samples from the just-finished task.

        Fires *after* :meth:`on_task_end`, which means the buffer has
        already been processed by importance estimators.  We sample
        ``episodic_mem_per_task`` transitions from the agent's main
        replay buffer (whatever is currently in there) before the
        runner clears it for the new task.
        """
        self._lazy_init(agent)
        if task_idx == 0:
            return
        prev_task = task_idx - 1
        buf = getattr(agent, "replay_buffer", None)
        if buf is None or getattr(buf, "size", 0) < self.episodic_mem_per_task:
            return
        new_mem = buf.sample_batch(self.episodic_mem_per_task)
        # ``store_multiple`` expects numpy; replay buffers return
        # tf.Tensors so we convert here once at the boundary.
        new_mem_np = {
            k: v.numpy() for k, v in new_mem.items()
            if k in (
                "obs", "next_obs", "actions", "rewards", "done", "one_hot",
            )
        }
        try:
            self._episodic_memory.store_multiple(**new_mem_np)
        except AssertionError:
            # Episodic memory full — silently skip; AGEM still uses
            # whatever is already stored as the constraint set.
            agent.logger.log(
                f"[agem] Episodic memory full at task {prev_task}; "
                "constraint set frozen.",
                color="yellow",
            )

    def get_episodic_batch(
        self, agent: "BaseAgent", *, task_idx: int
    ) -> Optional[Dict[str, tf.Tensor]]:
        """Return one mini-batch from the episodic memory, or ``None``.

        Returns ``None`` on the very first task (no past data) and
        whenever the memory is empty (e.g. when the agent's main
        replay buffer didn't have enough samples to refill it).
        """
        if task_idx == 0 or self._episodic_memory is None:
            return None
        if self._episodic_memory.size == 0:
            return None
        return self._episodic_memory.sample_batch(self.episodic_batch_size)

    def adjust_gradients(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Dict[str, List[Optional[tf.Tensor]]]:
        """Project current gradients onto the AGEM-feasible half-space.

        Returns the input dict unchanged on task 0 or when no
        episodic batch is available.  Otherwise computes per-group
        reference gradients via
        :meth:`BaseAgent.compute_reference_gradients` and projects
        each affected group separately.

        The projection is per-group rather than over the concatenated
        gradient — this matches the legacy SAC implementation, where
        actor and critic projections are independent.  For DQN there
        is only one group so the per-group form coincides with the
        original A-GEM formulation.
        """
        if task_idx == 0 or episodic_batch is None:
            return grads_by_group

        ref_grads_by_group = agent.compute_reference_gradients(episodic_batch)

        out: Dict[str, List[Optional[tf.Tensor]]] = dict(grads_by_group)
        for gn in self._regularize_groups:
            if gn not in grads_by_group or gn not in ref_grads_by_group:
                continue
            cur = grads_by_group[gn]
            ref = ref_grads_by_group[gn]
            out[gn] = self._project_group(cur, ref)
        return out

    @staticmethod
    def _project_group(
        new_grads: List[Optional[tf.Tensor]],
        ref_grads: List[Optional[tf.Tensor]],
    ) -> List[Optional[tf.Tensor]]:
        """Project ``new_grads`` onto the half-space ``⟨g, ref⟩ ≥ 0``.

        Computed inside the agent's @tf.function trace using
        ``tf.cond`` so the branch lives in the graph rather than at
        Python level.  Per-parameter ``None`` slots (typically rare —
        a parameter that doesn't influence the loss) are passed
        through untouched on both sides.
        """
        # Per-tensor sums of element-wise products → contribution to
        # the global dot product / squared norm.
        dot_prod = tf.zeros([])
        ref_sq_norm = tf.zeros([])
        for g, r in zip(new_grads, ref_grads):
            if g is None or r is None:
                continue
            dot_prod += tf.reduce_sum(g * r)
            ref_sq_norm += tf.reduce_sum(r * r)

        # Avoid division-by-zero when the reference gradient happens
        # to be exactly zero (e.g. an early-task parameter that the
        # episodic batch doesn't activate).
        ref_sq_norm = tf.maximum(ref_sq_norm, 1e-12)
        scale = dot_prod / ref_sq_norm

        projected: List[Optional[tf.Tensor]] = []
        for g, r in zip(new_grads, ref_grads):
            if g is None:
                projected.append(None)
                continue
            if r is None:
                projected.append(g)
                continue
            projected.append(
                tf.cond(dot_prod >= 0, lambda g=g: g, lambda g=g, r=r: g - scale * r)
            )
        return projected

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--episodic_mem_per_task",
            type=int,
            default=1000,
            help="Transitions stored per task in the episodic memory.",
        )
        parser.add_argument(
            "--episodic_batch_size",
            type=int,
            default=128,
            help="Mini-batch size for the AGEM reference gradient.",
        )
        parser.add_argument(
            "--regularize_groups",
            type=str,
            default=None,
            help=(
                "Comma-separated parameter-group names to project. "
                "Defaults to all groups."
            ),
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, agent: "BaseAgent"
    ) -> "AGEM":
        groups = (
            [g.strip() for g in args.regularize_groups.split(",") if g.strip()]
            if getattr(args, "regularize_groups", None)
            else None
        )
        return cls(
            episodic_mem_per_task=getattr(args, "episodic_mem_per_task", 1000),
            episodic_batch_size=getattr(args, "episodic_batch_size", 128),
            regularize_groups=groups,
        )
