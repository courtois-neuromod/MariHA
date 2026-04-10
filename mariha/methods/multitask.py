"""Multi-task joint training (continual learning upper bound).

Trains on all tasks simultaneously by never clearing the replay buffer
at task boundaries.  This is the standard joint-training upper bound
for continual-learning benchmarks: it ignores the sequential constraint
and lets data from every past task accumulate in a single buffer.

Implementation note
-------------------
There is no per-step or per-task algorithmic work for joint training —
the only thing that distinguishes it from a vanilla agent run is that
the replay buffer is never cleared.  In the composition refactor that
becomes a one-line agent flag (``reset_buffer_on_task_change=False``),
so this :class:`CLMethod` exists primarily as a name in the CL registry
that can be selected from the CLI.  At construction it asserts the
agent is configured for joint training and forces the flag if needed.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from mariha.benchmark.cl_registry import register_cl
from mariha.methods.base import CLMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


@register_cl("multitask")
class MultiTask(CLMethod):
    """Joint training over a shared replay buffer.

    Holds no state and overrides no per-step hooks; the only work it
    does is to set ``agent.reset_buffer_on_task_change = False`` on
    :meth:`on_task_start` so the buffer accumulates across tasks.
    Selecting this method from the CLI is equivalent to passing
    ``--reset_buffer_on_task_change=False`` to the agent.
    """

    name = "multitask"

    def on_task_start(self, agent: "BaseAgent", task_idx: int) -> None:
        """Force ``reset_buffer_on_task_change=False`` on the host agent.

        Set on every task boundary so that even if some other code path
        flips the flag back on between tasks, joint training still
        wins.  Logged once on the first task to make the override
        visible in run logs.
        """
        if getattr(agent, "reset_buffer_on_task_change", False):
            if task_idx == 0:
                agent.logger.log(
                    "[multitask] Forcing reset_buffer_on_task_change=False "
                    "(joint training).",
                    color="cyan",
                )
            agent.reset_buffer_on_task_change = False

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Multitask has no hyperparameters of its own."""

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, agent: "BaseAgent"
    ) -> "MultiTask":
        return cls()
