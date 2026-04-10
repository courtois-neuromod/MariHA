"""Standard checkpoint directory layout for MariHA agents.

Every agent writes task checkpoints under::

    {experiment_dir}/checkpoints/{agent_name}/{timestamp}_task{task_idx}/

This helper centralises the path construction so that all agents agree on
the convention and ``mariha-evaluate`` can discover checkpoints predictably.
"""

from __future__ import annotations

from pathlib import Path


def standard_checkpoint_dir(
    experiment_dir: Path,
    agent_name: str,
    timestamp: str,
    task_idx: int,
) -> Path:
    """Return the canonical per-task checkpoint directory.

    Args:
        experiment_dir: Root directory for the experiment (typically the
            parent of ``checkpoints/``).
        agent_name: Short agent identifier (e.g. ``"sac"``, ``"dqn"``).
        timestamp: Run timestamp string (empty string allowed for tests).
        task_idx: Zero-based index of the task within the curriculum.

    Returns:
        A :class:`Path` pointing to ``{experiment_dir}/checkpoints/{agent_name}/
        {timestamp}_task{task_idx}``.  The directory is *not* created here —
        callers should ``mkdir(parents=True, exist_ok=True)`` before writing.
    """
    ts = timestamp or ""
    return (
        Path(experiment_dir)
        / "checkpoints"
        / agent_name
        / f"{ts}_task{task_idx}"
    )
