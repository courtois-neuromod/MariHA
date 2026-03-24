"""EpisodeSpec dataclass: describes a single training episode.

Each ``EpisodeSpec`` corresponds to one human clip — one attempt by one
subject at one scene.  The curriculum loader (``loader.py``) constructs
these from the BIDS events TSV files in ``mario.scenes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeSpec:
    """Complete specification for a single training episode.

    Attributes:
        state_file: Absolute path to the gzip-compressed ``.state`` file
            that initialises the emulator at the start of this human clip.
        max_steps: Maximum number of environment steps allowed (equal to
            ``frame_stop - frame_start`` from the events TSV).
        scene_id: Scene identifier (e.g. ``'w1l1s0'``).
        clip_code: 14-character ordinal sort key derived from the replay
            filename (lower value = earlier in the session).
        subject: Subject identifier (e.g. ``'sub-01'``).
        session: Session identifier (e.g. ``'ses-001'``).
        outcome: Human outcome for this clip (``'completed'`` or ``'failed'``).
        phase: Experimental phase label (e.g. ``'discovery'``, ``'repetition'``).
        level: Level string (e.g. ``'w1l1'``).
        metadata: All remaining fields from the events TSV row as a dict
            (score, coins, enemies_killed, etc.).
    """

    state_file: Path
    max_steps: int
    scene_id: str
    clip_code: str
    subject: str
    session: str
    outcome: str
    phase: str
    level: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state_file = Path(self.state_file)
        self.max_steps = max(1, int(self.max_steps))
