"""Evaluation utilities for MariHA.

- :mod:`mariha.eval.metrics` — CL metric formulas (AP, BWT, forgetting, plasticity)
  and behavioral metric aggregation.
- :mod:`mariha.eval.runner` — run a trained SAC agent on a scene, discover checkpoints.

See ``scripts/evaluate.py`` for the full CLI evaluation pipeline.
"""

from mariha.eval.metrics import (
    aggregate_behavioral_stats,
    compute_cl_metrics,
    summarise_behavioral_metrics,
)
from mariha.eval.runner import eval_on_scene, find_task_checkpoints

__all__ = [
    "aggregate_behavioral_stats",
    "compute_cl_metrics",
    "eval_on_scene",
    "find_task_checkpoints",
    "summarise_behavioral_metrics",
]
