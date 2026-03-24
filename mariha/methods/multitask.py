"""Multi-task SAC (joint training upper bound).

Trains on all tasks simultaneously by never clearing the replay buffer at
task boundaries.  This is the standard joint-training upper bound for CL
benchmarks — it ignores the sequential constraint and has access to data
from all past tasks throughout training.

Usage: identical to ``SAC``; just set ``reset_buffer_on_task_change=False``
(enforced by default in this subclass).
"""

from __future__ import annotations

from mariha.rl.sac import SAC


class MultiTask_SAC(SAC):
    """SAC with a shared, never-cleared replay buffer (joint training).

    Overrides ``reset_buffer_on_task_change`` to ``False`` regardless of the
    value passed in ``kwargs``, so data from all past tasks accumulates in a
    single buffer and is available at every gradient step.

    All other arguments are forwarded unchanged to
    :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(self, **kwargs) -> None:
        # Force shared buffer — the defining property of joint training.
        kwargs["reset_buffer_on_task_change"] = False
        super().__init__(**kwargs)
