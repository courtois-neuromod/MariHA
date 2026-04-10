"""ClonEx (Clone-and-Explore) continual learning baseline.

Extends DER with critic-side distillation: in addition to the
KL-divergence anchor on actor logits / Q-values, ClonEx adds an MSE
penalty between the *stored* and *current* critic outputs.

* On SAC, both ``critic1_q`` and ``critic2_q`` are distilled.
* On PPO, the scalar ``value`` head is distilled.
* On DQN, ``clone_critic`` is a no-op — the single ``q_values`` target
  already provides the only available signal.

Reference: Ben-Iwhiwhu et al., 2021 — *Lifelong Robotic Reinforcement
Learning by Retaining Experiences*, arXiv:2105.07748.

ClonEx is a thin :class:`DistillationMethod` subclass that flips
``clone_critic=True`` and exposes the additional ``critic_alpha``
hyperparameter; everything else (episodic memory snapshot, gradient
injection, agent dispatch) is shared with :class:`DER`.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.distillation_base import DistillationMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("clonex")
class ClonEx(DistillationMethod):
    """ClonEx — actor + critic distillation from episodic memory.

    Args:
        episodic_mem_per_task: Transitions stored per completed task.
        episodic_batch_size: Mini-batch sampled from episodic memory at
            each gradient step.
        actor_alpha: Weight of the actor / Q-value distillation loss.
        critic_alpha: Weight of the critic-side MSE distillation loss.
            Ignored on DQN (no separate critic).
        temperature: Temperature ``T`` for the soft Q-value KL on DQN.
    """

    name = "clonex"

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        actor_alpha: float = 0.1,
        critic_alpha: float = 0.1,
        temperature: float = 2.0,
    ) -> None:
        super().__init__(
            episodic_mem_per_task=episodic_mem_per_task,
            episodic_batch_size=episodic_batch_size,
            actor_alpha=actor_alpha,
            critic_alpha=critic_alpha,
            clone_critic=True,
            temperature=temperature,
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, agent: "BaseAgent"
    ) -> "ClonEx":
        return cls(
            episodic_mem_per_task=getattr(args, "episodic_mem_per_task", 1000),
            episodic_batch_size=getattr(args, "episodic_batch_size", 128),
            actor_alpha=getattr(args, "actor_alpha", 0.1),
            critic_alpha=getattr(args, "critic_alpha", 0.1),
            temperature=getattr(args, "distill_temperature", 2.0),
        )
