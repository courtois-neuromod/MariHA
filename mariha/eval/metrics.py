"""Continual learning metrics for the MariHA benchmark.

Standard CL metrics follow the notation of Lopez-Paz & Ranzato (2017) and
Riemer et al. (2018):

- **AP**  — Average Performance: mean return across all scenes with the
  final model.  The primary summary metric.
- **BWT** — Backward Transfer: how much the final model's performance on
  past tasks differs from its performance right after learning each task.
  Negative BWT indicates catastrophic forgetting.
- **Forgetting** — ``max(0, -BWT)``: unsigned forgetting score.
- **Plasticity** — performance on the last task only (ability to learn
  new tasks without interference from the past).

Additionally, MariHA-specific behavioral metrics are derived from
``EpisodeStats``:

- **clear_rate** — fraction of episodes that reached the exit point.
- **mean_x_traveled** — mean X-distance covered per episode.
- **mean_score_gained** — mean score delta per episode.
- **death_rate** — fraction of episodes ending in death.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


import numpy as np


def compute_cl_metrics(
    R_final: np.ndarray,
    R_diag: Optional[np.ndarray] = None,
    scene_ids: Optional[Sequence[str]] = None,
) -> Dict:
    """Compute CL metrics from performance vectors.

    Args:
        R_final: Shape ``(N,)`` — return on each scene with the *final*
            checkpoint (after all tasks).
        R_diag: Shape ``(N,)`` — return on scene ``k`` with the
            *task-k* checkpoint (right after training task ``k``).
            Required for BWT / forgetting.
        scene_ids: Optional ordered list of scene IDs for labelling.

    Returns:
        Dictionary of computed metrics.  Keys:

        - ``AP`` — mean of ``R_final``.
        - ``AP_std`` — standard deviation of ``R_final``.
        - ``per_scene_final`` — per-scene dict (or list if no scene_ids).
        - ``BWT`` — backward transfer (only if ``R_diag`` provided).
        - ``forgetting`` — ``max(0, -BWT)`` (only if ``R_diag`` provided).
        - ``plasticity`` — ``R_diag[-1]`` (only if ``R_diag`` provided).
        - ``per_scene_peak`` — per-scene peak performance dict
          (only if ``R_diag`` provided).
    """
    R_final = np.asarray(R_final, dtype=np.float64)
    N = len(R_final)

    per_scene_final = (
        {scene_ids[i]: float(R_final[i]) for i in range(N)}
        if scene_ids is not None
        else list(map(float, R_final))
    )

    metrics: Dict = {
        "AP": float(np.mean(R_final)),
        "AP_std": float(np.std(R_final)),
        "per_scene_final": per_scene_final,
    }

    if R_diag is not None:
        R_diag = np.asarray(R_diag, dtype=np.float64)
        # Exclude the last task from BWT (no "past" performance to compare).
        if N > 1:
            bwt = float(np.mean(R_final[:-1] - R_diag[:-1]))
        else:
            bwt = 0.0
        metrics["BWT"] = bwt
        metrics["forgetting"] = float(max(0.0, -bwt))
        metrics["plasticity"] = float(R_diag[-1])
        per_scene_peak = (
            {scene_ids[i]: float(R_diag[i]) for i in range(N)}
            if scene_ids is not None
            else list(map(float, R_diag))
        )
        metrics["per_scene_peak"] = per_scene_peak

    return metrics


def aggregate_behavioral_stats(
    stats_list: List[Dict],
) -> Dict:
    """Aggregate a list of ``EpisodeStats.to_dict()`` outputs.

    Args:
        stats_list: List of dicts from ``EpisodeStats.to_dict()``.
            Must contain keys: ``cleared``, ``x_traveled``,
            ``score_gained``, ``lives_lost``.

    Returns:
        Summary dict with ``clear_rate``, ``mean_x_traveled``,
        ``mean_score_gained``, ``death_rate``.
    """
    if not stats_list:
        return {}
    return {
        "clear_rate": float(np.mean([s.get("cleared", False) for s in stats_list])),
        "mean_x_traveled": float(np.mean([s.get("x_traveled", 0) for s in stats_list])),
        "mean_score_gained": float(np.mean([s.get("score_gained", 0) for s in stats_list])),
        "death_rate": float(np.mean([s.get("lives_lost", 0) for s in stats_list])),
    }


def summarise_behavioral_metrics(
    per_scene_behavioral: Dict[str, Dict],
) -> Dict:
    """Average per-scene behavioral dicts into a single summary.

    Args:
        per_scene_behavioral: ``scene_id`` → dict from
            :func:`aggregate_behavioral_stats`.

    Returns:
        Mean across scenes for each behavioral key.
    """
    if not per_scene_behavioral:
        return {}
    keys = list(next(iter(per_scene_behavioral.values())).keys())
    return {
        k: float(np.mean([v[k] for v in per_scene_behavioral.values() if k in v]))
        for k in keys
    }
