"""Dark Experience Replay (DER) continual learning baseline.

At the boundary of every task, DER snapshots a fixed-size batch of
transitions from the just-finished task's replay buffer together with
the agent's current distillation targets (actor logits for SAC/PPO,
Q-values for DQN).  During subsequent tasks a KL-divergence loss
anchors the *current* distillation outputs to the *stored* ones,
preventing the parameters that produced past behaviour from drifting.

DER is the actor-only variant: only the policy / Q-value head receives
a distillation gradient.  ClonEx (:mod:`mariha.methods.clonex`) extends
it with critic-side distillation.

Reference: Buzzega et al., 2020 — *Dark Experience for General Continual
Learning: a Strong, Simple Baseline*, arXiv:2004.07211.

Agent dispatch
--------------
The base class :class:`DistillationMethod` inspects the keys of
``agent.distill_targets()`` to choose the loss formula:

* SAC/PPO return ``"actor_logits"`` → softmax KL.
* DQN returns ``"q_values"`` → soft KL with temperature ``T``
  (Rusu et al. 2016, *Policy Distillation*).

DER does not require any agent-specific code paths.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.distillation_base import DistillationMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("der")
class DER(DistillationMethod):
    """Dark Experience Replay — actor-only distillation.

    A thin :class:`DistillationMethod` subclass that pins
    ``clone_critic=False``.  All bookkeeping (episodic memory snapshot,
    per-step replay batch sampling, gradient injection) lives in the
    base class.

    Args:
        episodic_mem_per_task: Transitions stored per completed task.
        episodic_batch_size: Mini-batch sampled from episodic memory at
            each gradient step.
        actor_alpha: Weight of the KL distillation loss.
        temperature: Temperature ``T`` for the soft Q-value KL on DQN
            (Rusu 2016).  Ignored on SAC/PPO, which use plain softmax
            KL on actor logits.
    """

    name = "der"

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        actor_alpha: float = 0.1,
        temperature: float = 2.0,
    ) -> None:
        super().__init__(
            episodic_mem_per_task=episodic_mem_per_task,
            episodic_batch_size=episodic_batch_size,
            actor_alpha=actor_alpha,
            critic_alpha=0.0,
            clone_critic=False,
            temperature=temperature,
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, agent: "BaseAgent"
    ) -> "DER":
        return cls(
            episodic_mem_per_task=getattr(args, "episodic_mem_per_task", 1000),
            episodic_batch_size=getattr(args, "episodic_batch_size", 128),
            actor_alpha=getattr(args, "actor_alpha", 0.1),
            temperature=getattr(args, "distill_temperature", 2.0),
        )
