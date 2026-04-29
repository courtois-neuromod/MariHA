"""Curriculum metadata helpers for the dense eval matrix.

Under the run-as-task model, classical CL metrics (AP / BWT / forgetting)
partially lose their meaning: scenes overlap across tasks (a scene first
seen in discovery is revisited in practice runs), so a scene's return at
any single checkpoint reflects all of its prior exposures, not a single
consolidation point.

The eval pipeline therefore emits a *dense* return matrix
(``returns_matrix[run_index][scene_id]``) and lets analysis notebooks
derive whatever metric makes sense (per-scene learning curves,
first-exposure-vs-final, discovery-vs-practice contrasts, etc.).

This module exposes the per-scene and per-run metadata needed to
interpret that matrix.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable

from mariha.curriculum.episode import EpisodeSpec


def build_scene_metadata(
    sequence: Iterable[EpisodeSpec],
) -> Dict[str, Dict]:
    """Per-scene metadata derived from the curriculum.

    Args:
        sequence: An iterable of ``EpisodeSpec`` (e.g. a ``HumanSequence``).

    Returns:
        ``{scene_id: {first_exposure_run_index, level, n_clips_in_curriculum}}``.
    """
    out: Dict[str, Dict] = {}
    counts: Counter = Counter()
    for spec in sequence:
        counts[spec.scene_id] += 1
        if spec.scene_id not in out:
            out[spec.scene_id] = {
                "first_exposure_run_index": int(spec.run_index),
                "level": spec.level,
                "n_clips_in_curriculum": 0,
            }
    for scene_id, n in counts.items():
        out[scene_id]["n_clips_in_curriculum"] = int(n)
    return out


def build_run_metadata(
    sequence: Iterable[EpisodeSpec],
) -> Dict[int, Dict]:
    """Per-run metadata derived from the curriculum.

    Args:
        sequence: An iterable of ``EpisodeSpec``.

    Returns:
        ``{run_index: {run_id, session, phase, levels, n_scenes, n_clips}}``.
        ``levels`` is a sorted list of unique level strings encountered in
        the run; ``phase`` is the modal phase label (typically constant
        within a run).
    """
    by_run: Dict[int, Dict] = {}
    for spec in sequence:
        ri = int(spec.run_index)
        if ri not in by_run:
            by_run[ri] = {
                "run_id": spec.run_id,
                "session": spec.session,
                "phases": Counter(),
                "levels": set(),
                "scenes": set(),
                "n_clips": 0,
            }
        entry = by_run[ri]
        entry["phases"][spec.phase] += 1
        entry["levels"].add(spec.level)
        entry["scenes"].add(spec.scene_id)
        entry["n_clips"] += 1

    out: Dict[int, Dict] = {}
    for ri, entry in by_run.items():
        modal_phase = entry["phases"].most_common(1)[0][0]
        out[ri] = {
            "run_id": entry["run_id"],
            "session": entry["session"],
            "phase": modal_phase,
            "levels": sorted(entry["levels"]),
            "n_scenes": len(entry["scenes"]),
            "n_clips": entry["n_clips"],
        }
    return out
