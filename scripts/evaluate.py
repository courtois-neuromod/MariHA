"""Evaluation script for MariHA continual learning runs.

Loads a trained agent's checkpoint(s) and evaluates it on the training
scenes, then computes standard CL metrics.

Usage::

    python scripts/evaluate.py \\
        --subject sub-01 \\
        --agent sac \\
        --cl_method ewc \\
        --run_prefix 20260322_120000_seed0 \\
        --n_episodes 5

Or with the installed entry point::

    mariha-evaluate --subject sub-01 --agent sac --cl_method ewc --run_prefix ...

The ``--cl_method`` flag is optional; omit it to evaluate a vanilla
agent run.

Outputs a ``results.json`` in the run directory containing:

- CL metrics (AP, BWT, forgetting, plasticity) if ``--eval_diagonal`` is set
- Per-scene returns and behavioral stats
- Metadata (method, subject, n_episodes, ...)

Eval protocol
-------------
"Same scenes as training": for each unique scene in the subject's
curriculum, the FIRST ``EpisodeSpec`` is used as the canonical eval
episode (same starting state file as training).

Two eval modes
--------------
``final`` (default):
    Load the final checkpoint (highest task index found), evaluate on
    all training scenes.  Computes AP and behavioral metrics.

``diagonal`` (set ``--eval_diagonal``):
    Additionally loads the per-task checkpoint for each scene and
    evaluates it on that scene.  Enables BWT and forgetting computation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from mariha.curriculum.sequences import HumanSequence
from mariha.env.base import SCENARIOS_DIR
from mariha.env.scenario_gen import load_metadata
from mariha.eval.metrics import (
    aggregate_behavioral_stats,
    compute_cl_metrics,
    summarise_behavioral_metrics,
)
from mariha.eval.runner import eval_on_scene, find_task_checkpoints
import mariha.rl  # noqa: F401 — registers all built-in agents
from mariha.benchmark.registry import get_agent_class
from mariha.utils.logging import EpochLogger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained MariHA agent on its training scenes."
    )
    p.add_argument("--subject", required=True, help="Subject ID, e.g. sub-01.")
    p.add_argument(
        "--agent",
        default="sac",
        help="Agent name (from registry).",
    )
    p.add_argument(
        "--cl_method",
        default=None,
        help="Continual learning method composed during training "
        "(must match the value passed to mariha-run-cl). Omit for "
        "vanilla agent runs.",
    )
    p.add_argument(
        "--run_prefix",
        required=True,
        help="Run prefix used when training, e.g. '20260322_120000_seed0'. "
        "Used to locate checkpoint directories.",
    )
    p.add_argument(
        "--experiment_dir",
        default="experiments",
        help="Root experiments directory (default: experiments).",
    )
    p.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="Episodes per scene during evaluation (default: 5).",
    )
    p.add_argument(
        "--eval_diagonal",
        action="store_true",
        help="Also evaluate per-task checkpoints on their own scene "
        "(enables BWT and forgetting metrics; slower).",
    )
    p.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Evaluate only the first N scenes (for quick debugging).",
    )
    p.add_argument("--seed", type=int, default=0)
    # SAC-specific network flags kept for backwards compat; ignored by other agents
    p.add_argument("--hidden_sizes", nargs="+", type=int, default=[256, 256])
    p.add_argument("--activation", default="tanh")
    p.add_argument("--use_layer_norm", action="store_true")
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--hide_task_id", action="store_true")
    p.add_argument("--render_mode", default=None)
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: auto-derived from experiment_dir/run_prefix).",
    )
    return p


# ---------------------------------------------------------------------------
# Agent construction via registry
# ---------------------------------------------------------------------------


def _build_eval_agent(agent_name: str, checkpoint_dir: Path, env, scene_ids, args):
    """Instantiate an agent from the registry and load a checkpoint."""
    agent_cls = get_agent_class(agent_name)
    eval_logger = EpochLogger(
        output_dir="/tmp/mariha_eval_tmp",
        logger_output=["stdout"],
    )
    agent = agent_cls.from_args(args, env=env, logger=eval_logger, scene_ids=scene_ids)
    agent.load_checkpoint(checkpoint_dir)
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_eval_parser()
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    agent_name = args.agent
    cl_name = args.cl_method
    # Composite label used by ``run_cl.py`` / ``run_single.py`` after CL
    # composition: bare agent name for vanilla runs, ``{agent}_{cl}`` for
    # CL-augmented runs.  Checkpoint and output dirs are keyed off this so
    # different CL methods on the same base agent never collide.
    run_label = f"{agent_name}_{cl_name}" if cl_name else agent_name
    checkpoint_base = experiment_dir / "checkpoints" / run_label

    # ------------------------------------------------------------------ #
    # Discover checkpoints
    # ------------------------------------------------------------------ #
    task_checkpoints = find_task_checkpoints(checkpoint_base, args.run_prefix)
    if not task_checkpoints:
        logger.error(
            "No checkpoints found in %s matching prefix '%s'.",
            checkpoint_base, args.run_prefix,
        )
        sys.exit(1)

    final_task_idx = max(task_checkpoints.keys())
    final_checkpoint = task_checkpoints[final_task_idx]
    logger.info(
        "Found %d checkpoint(s). Final: task%d (%s)",
        len(task_checkpoints), final_task_idx, final_checkpoint,
    )

    # ------------------------------------------------------------------ #
    # Curriculum: one eval spec per scene (first clip per scene)
    # ------------------------------------------------------------------ #
    sequence = HumanSequence(subject_id=args.subject)
    eval_specs: dict = {}
    for spec in sequence:
        if spec.scene_id not in eval_specs:
            eval_specs[spec.scene_id] = spec

    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())

    # Determine which scenes to evaluate (those seen during training).
    scenes_to_eval = [s for s in scene_ids if s in eval_specs]
    if args.max_scenes is not None:
        scenes_to_eval = scenes_to_eval[: args.max_scenes]
    logger.info("Evaluating on %d scenes.", len(scenes_to_eval))

    # ------------------------------------------------------------------ #
    # Build a minimal dummy env to get obs/action spaces for agent init
    # ------------------------------------------------------------------ #
    from mariha.env.continual import make_scene_env

    first_scene = scenes_to_eval[0]
    dummy_env = make_scene_env(
        scene_id=first_scene,
        exit_point=scene_meta[first_scene]["exit_point"],
        scene_ids=scene_ids,
    )

    # ------------------------------------------------------------------ #
    # Final-checkpoint eval (all scenes)
    # ------------------------------------------------------------------ #
    logger.info("=== Final checkpoint eval ===")
    final_agent = _build_eval_agent(
        agent_name, final_checkpoint, dummy_env, scene_ids, args
    )

    R_final = np.zeros(len(scenes_to_eval))
    per_scene_behavioral: dict = {}

    for i, scene_id in enumerate(scenes_to_eval):
        spec = eval_specs[scene_id]
        mean_ret, stats = eval_on_scene(
            final_agent, scene_id, spec, scene_ids,
            n_episodes=args.n_episodes,
            scenarios_dir=SCENARIOS_DIR,
        )
        R_final[i] = mean_ret
        per_scene_behavioral[scene_id] = aggregate_behavioral_stats(stats)
        logger.info("  [%3d/%3d] %s  return=%.2f  clear=%.0f%%",
                    i + 1, len(scenes_to_eval), scene_id,
                    mean_ret, per_scene_behavioral[scene_id].get("clear_rate", 0) * 100)

    # ------------------------------------------------------------------ #
    # Diagonal eval (optional — loads per-task checkpoint per scene)
    # ------------------------------------------------------------------ #
    R_diag: np.ndarray | None = None

    if args.eval_diagonal:
        logger.info("=== Diagonal eval (per-task checkpoints) ===")
        R_diag = np.zeros(len(scenes_to_eval))

        for i, scene_id in enumerate(scenes_to_eval):
            task_idx = scene_ids.index(scene_id)
            if task_idx not in task_checkpoints:
                logger.warning(
                    "No checkpoint for task %d (%s); using final instead.",
                    task_idx, scene_id,
                )
                R_diag[i] = R_final[i]
                continue

            task_agent = _build_eval_agent(
                agent_name, task_checkpoints[task_idx],
                dummy_env, scene_ids, args,
            )
            spec = eval_specs[scene_id]
            mean_ret, _ = eval_on_scene(
                task_agent, scene_id, spec, scene_ids,
                n_episodes=args.n_episodes,
                scenarios_dir=SCENARIOS_DIR,
            )
            R_diag[i] = mean_ret
            logger.info(
                "  [%3d/%3d] %s  diag_return=%.2f",
                i + 1, len(scenes_to_eval), scene_id, mean_ret,
            )

    dummy_env.close()

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    cl_metrics = compute_cl_metrics(
        R_final=R_final,
        R_diag=R_diag,
        scene_ids=scenes_to_eval,
    )
    behavioral_summary = summarise_behavioral_metrics(per_scene_behavioral)

    results = {
        "metadata": {
            "subject": args.subject,
            "agent": agent_name,
            "cl_method": cl_name,
            "run_label": run_label,
            "run_prefix": args.run_prefix,
            "n_episodes": args.n_episodes,
            "n_scenes": len(scenes_to_eval),
            "eval_diagonal": args.eval_diagonal,
        },
        "cl_metrics": cl_metrics,
        "behavioral_summary": behavioral_summary,
        "per_scene_behavioral": per_scene_behavioral,
    }

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if args.output:
        out_path = Path(args.output)
    else:
        run_dir = (
            experiment_dir / args.subject / run_label / args.run_prefix
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "eval_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Results saved to %s", out_path)
    logger.info(
        "AP=%.3f%s",
        cl_metrics["AP"],
        f"  BWT={cl_metrics['BWT']:.3f}  Forgetting={cl_metrics['forgetting']:.3f}"
        if "BWT" in cl_metrics else "",
    )


if __name__ == "__main__":
    main()
