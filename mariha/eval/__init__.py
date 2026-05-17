"""Evaluation utilities for MariHA.

- :mod:`mariha.eval.metrics` — CL metric formulas (AP, BWT, forgetting, plasticity)
  and behavioral metric aggregation.
- :mod:`mariha.eval.runner` — run a trained SAC agent on a scene, discover checkpoints.

See ``scripts/evaluate.py`` for the full CLI evaluation pipeline.
"""

from mariha.eval.metrics import build_run_metadata, build_scene_metadata
from mariha.eval.runner import eval_on_scene, find_task_checkpoints

__all__ = [
    "build_run_metadata",
    "build_scene_metadata",
    "eval_on_scene",
    "find_task_checkpoints",
]
