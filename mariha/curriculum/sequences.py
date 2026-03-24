"""Sequence providers for the continual learning curriculum.

A sequence is any iterable of ``EpisodeSpec`` objects.  The training loop
consumes one spec per episode, in the order provided.

The default sequence is ``HumanSequence``, which replays the human gameplay
data for a single subject in chronological order (sorted by ``clip_code``).

Future sequence types (e.g. ``LevelSequence``, ``PatternSequence``) can be
implemented by subclassing ``BaseSequence`` and overriding ``_build()``.

Usage::

    from mariha.curriculum.sequences import HumanSequence

    seq = HumanSequence(subject_id="sub-01")
    for spec in seq:
        ...  # one EpisodeSpec per iteration
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from mariha.curriculum.episode import EpisodeSpec
from mariha.curriculum.loader import _SCENES_ROOT, load_curriculum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSequence(ABC):
    """Abstract base class for curriculum sequences.

    Subclasses must implement ``_build()`` to return an ordered list of
    ``EpisodeSpec`` objects.  The base class provides ``__iter__``,
    ``__len__``, and a ``scene_ids`` property for convenience.
    """

    def __init__(self) -> None:
        self._specs: list[EpisodeSpec] = self._build()

    @abstractmethod
    def _build(self) -> list[EpisodeSpec]:
        """Build and return the ordered list of episode specs."""

    def __iter__(self) -> Iterator[EpisodeSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def scene_ids(self) -> list[str]:
        """Ordered unique scene IDs encountered in this sequence."""
        seen: dict[str, None] = {}
        for spec in self._specs:
            seen[spec.scene_id] = None
        return list(seen)

    @property
    def subject(self) -> str | None:
        """Subject ID if all specs share the same subject, else ``None``."""
        subjects = {s.subject for s in self._specs}
        return next(iter(subjects)) if len(subjects) == 1 else None


# ---------------------------------------------------------------------------
# Human-aligned sequence (default)
# ---------------------------------------------------------------------------


class HumanSequence(BaseSequence):
    """Human-aligned episode sequence for a single subject.

    Episodes are ordered by ``clip_code`` (ascending), matching the
    temporal order in which the human encountered each scene.

    Args:
        subject_id: Subject identifier (e.g. ``'sub-01'``).
        scenes_root: Root of the ``mario.scenes`` dataset.
        require_existing_states: Skip clips with missing ``.state`` files
            when ``True`` (default).
    """

    def __init__(
        self,
        subject_id: str,
        scenes_root: Path = _SCENES_ROOT,
        require_existing_states: bool = True,
    ) -> None:
        self._subject_id = subject_id
        self._scenes_root = scenes_root
        self._require_existing_states = require_existing_states
        super().__init__()

    def _build(self) -> list[EpisodeSpec]:
        return load_curriculum(
            subject_id=self._subject_id,
            scenes_root=self._scenes_root,
            require_existing_states=self._require_existing_states,
        )


# ---------------------------------------------------------------------------
# Future sequence stubs (not implemented in MVP)
# ---------------------------------------------------------------------------

# These classes are placeholders that document the intended interface for
# synthetic sequences.  Raise NotImplementedError to surface clearly if
# accidentally used before implementation.


class LevelSequence(BaseSequence):
    """Sequence filtered to a single Mario level (future).

    Args:
        subject_id: Subject identifier.
        level: Level string (e.g. ``'w1l1'``).
    """

    def __init__(self, subject_id: str, level: str) -> None:
        self._subject_id = subject_id
        self._level = level
        super().__init__()

    def _build(self) -> list[EpisodeSpec]:
        raise NotImplementedError("LevelSequence is not yet implemented.")


class PatternSequence(BaseSequence):
    """Sequence filtered by game-design pattern (future).

    Args:
        subject_id: Subject identifier.
        pattern: Pattern column name from ``scenes_mastersheet.csv``
            (e.g. ``'Gap'``, ``'Enemy'``).
    """

    def __init__(self, subject_id: str, pattern: str) -> None:
        self._subject_id = subject_id
        self._pattern = pattern
        super().__init__()

    def _build(self) -> list[EpisodeSpec]:
        raise NotImplementedError("PatternSequence is not yet implemented.")


class ExpandedBudgetSequence(BaseSequence):
    """Human sequence with the per-episode frame budget scaled up (future).

    Args:
        subject_id: Subject identifier.
        multiplier: Factor by which to multiply each clip's ``max_steps``.
    """

    def __init__(self, subject_id: str, multiplier: float) -> None:
        self._subject_id = subject_id
        self._multiplier = multiplier
        super().__init__()

    def _build(self) -> list[EpisodeSpec]:
        raise NotImplementedError("ExpandedBudgetSequence is not yet implemented.")
