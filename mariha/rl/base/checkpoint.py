"""Standard checkpoint directory layout for MariHA agents.

Every agent writes task checkpoints under::

    {experiment_dir}/checkpoints/{agent_name}/{subject}/{timestamp}_task{task_idx}/

The ``{subject}`` component is omitted when no subject is given (unit tests,
single-scene runs), so those keep the shorter ``.../{agent_name}/{timestamp}_
task{task_idx}`` layout.  This helper centralises the path construction so all
agents agree on the convention and ``mariha-evaluate`` can discover
checkpoints predictably.
"""

from __future__ import annotations

from pathlib import Path


def standard_checkpoint_dir(
    experiment_dir: Path,
    agent_name: str,
    timestamp: str,
    task_idx: int,
    subject: str = "",
) -> Path:
    """Return the canonical per-task checkpoint directory.

    Args:
        experiment_dir: Root directory for the experiment (typically the
            parent of ``checkpoints/``).
        agent_name: Short agent identifier (e.g. ``"sac"``, ``"dqn"``).
        timestamp: Run timestamp string (empty string allowed for tests).
        task_idx: Zero-based index of the task within the curriculum.
        subject: Subject ID (e.g. ``"sub-01"``).  When non-empty it is
            inserted as a directory level so checkpoints for different
            subjects of the same agent never collide.  Empty string keeps
            the legacy subject-less layout.

    Returns:
        A :class:`Path` pointing to ``{experiment_dir}/checkpoints/{agent_name}/
        [{subject}/]{timestamp}_task{task_idx}``.  The directory is *not*
        created here — callers should ``mkdir(parents=True, exist_ok=True)``
        before writing.
    """
    ts = timestamp or ""
    base = Path(experiment_dir) / "checkpoints" / agent_name
    if subject:
        base = base / subject
    return base / f"{ts}_task{task_idx}"
