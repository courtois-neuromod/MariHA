"""Shared infrastructure for MariHA RL agents.

This package contains the pieces that every algorithm (SAC, PPO, DQN, and
future additions like MuZero or LLM-based agents) can reuse:

- :class:`BaseAgent` — a :class:`BenchmarkAgent` subclass that implements
  ``run()`` once via the :class:`TrainingLoopRunner` and exposes a small
  callback surface for algorithm-specific work (``select_action``,
  ``store_transition``, ``should_update``, ``update_step``, ``save_weights``,
  ``load_weights``).
- :class:`TrainingLoopRunner` — the episode-driven training loop shared by
  per-step (off-policy) and per-rollout (on-policy) agents.
- ``run_burn_in`` — the shared pre-training driver for a single scene.
- ``standard_checkpoint_dir`` — the canonical ``experiments/checkpoints/...``
  path layout used across agents.

Concrete agents should subclass :class:`BaseAgent` and implement only the
algorithm-specific pieces; the boilerplate (episode handling, task switches,
session-boundary flushes, render checkpoints, logging, periodic saves) is
provided here.
"""

from mariha.rl.base.agent_base import BaseAgent
from mariha.rl.base.burn_in import run_burn_in
from mariha.rl.base.checkpoint import standard_checkpoint_dir
from mariha.rl.base.training_loop import TrainingLoopRunner

__all__ = [
    "BaseAgent",
    "TrainingLoopRunner",
    "run_burn_in",
    "standard_checkpoint_dir",
]
