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
from typing import Dict, Iterable, List, Optional

import numpy as np

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


def aggregate_behavioral_stats(stats_list: List[Dict]) -> Dict:
    """Aggregate per-episode behavioral dicts into scene-level means."""
    if not stats_list:
        return {}
    return {
        "clear_rate": float(np.mean([s.get("cleared", False) for s in stats_list])),
        "mean_x_traveled": float(np.mean([s.get("x_traveled", 0) for s in stats_list])),
        "mean_score_gained": float(np.mean([s.get("score_gained", 0) for s in stats_list])),
        "death_rate": float(np.mean([s.get("lives_lost", 0) for s in stats_list])),
    }


def compute_cl_metrics(
    returns_matrix: Dict[str, Dict[str, float]],
    scene_metadata: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Compute AP, BWT, and forgetting from the dense returns matrix.

    Args:
        returns_matrix: ``{str(run_index): {scene_id: mean_return}}``.
        scene_metadata: Optional output of ``build_scene_metadata``, used to
            look up each scene's first-exposure run index for BWT/forgetting.

    Returns:
        Dict with keys ``ap``, ``bwt``, ``forgetting``, ``n_scenes``,
        ``n_checkpoints``. BWT and forgetting are ``None`` when
        ``scene_metadata`` is not provided.
    """
    run_indices = sorted(int(k) for k in returns_matrix)
    if not run_indices:
        return {}

    final_returns = returns_matrix[str(max(run_indices))]
    ap = float(np.mean(list(final_returns.values()))) if final_returns else 0.0

    bwt_values: List[float] = []
    forgetting_values: List[float] = []

    if scene_metadata:
        for scene_id, r_final in final_returns.items():
            meta = scene_metadata.get(scene_id, {})
            first_ri = meta.get("first_exposure_run_index")
            if first_ri is None:
                continue
            row = returns_matrix.get(str(first_ri), {})
            r_first = row.get(scene_id)
            if r_first is None:
                continue
            delta = r_final - r_first
            bwt_values.append(delta)
            forgetting_values.append(max(0.0, -delta))

    return {
        "ap": ap,
        "bwt": float(np.mean(bwt_values)) if bwt_values else None,
        "forgetting": float(np.mean(forgetting_values)) if forgetting_values else None,
        "n_scenes": len(final_returns),
        "n_checkpoints": len(run_indices),
    }


def summarise_behavioral_metrics(
    behavioral_matrix: Dict[str, Dict[str, Dict]],
) -> Dict:
    """Aggregate behavioral stats across all (checkpoint, scene) cells."""
    clear_rates, x_traveled, scores, death_rates = [], [], [], []
    for run_stats in behavioral_matrix.values():
        for cell in run_stats.values():
            if not cell:
                continue
            clear_rates.append(cell.get("clear_rate", 0.0))
            x_traveled.append(cell.get("mean_x_traveled", 0.0))
            scores.append(cell.get("mean_score_gained", 0.0))
            death_rates.append(cell.get("death_rate", 0.0))

    def _mean(lst: list) -> Optional[float]:
        return float(np.mean(lst)) if lst else None

    return {
        "mean_clear_rate": _mean(clear_rates),
        "mean_x_traveled": _mean(x_traveled),
        "mean_score_gained": _mean(scores),
        "mean_death_rate": _mean(death_rates),
    }
