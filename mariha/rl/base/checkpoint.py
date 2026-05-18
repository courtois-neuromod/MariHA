"""Standard checkpoint directory layout for MariHA agents.

Every agent writes task checkpoints under::

    {checkpoint_dir}/checkpoints/{subject}/{agent_name}/{timestamp}_task{task_idx}/

This helper centralises the path construction so that all agents agree on
the convention and ``mariha-evaluate`` can discover checkpoints predictably.
"""

from __future__ import annotations

from pathlib import Path


def standard_checkpoint_dir(
    checkpoint_dir: Path,
    agent_name: str,
    timestamp: str,
    task_idx: int,
    subject: str = "",
) -> Path:
    """Return the canonical per-task checkpoint directory.

    Args:
        checkpoint_dir: Root directory for checkpoints.
        agent_name: Short agent identifier (e.g. ``"sac"``, ``"dqn"``).
        timestamp: Run timestamp string (empty string allowed for tests).
        task_idx: Zero-based index of the task within the curriculum.
        subject: Subject ID (e.g. ``"sub-01"``). Inserted between
            ``checkpoints/`` and ``agent_name/`` so runs from different
            subjects don't collide.

    Returns:
        A :class:`Path` pointing to ``{checkpoint_dir}/checkpoints/{subject}/
        {agent_name}/{timestamp}_task{task_idx}``.  The directory is *not*
        created here — callers should ``mkdir(parents=True, exist_ok=True)``
        before writing.
    """
    ts = timestamp or ""
    base = Path(checkpoint_dir) / "checkpoints"
    if subject:
        base = base / subject
    return base / agent_name / f"{ts}_task{task_idx}"
