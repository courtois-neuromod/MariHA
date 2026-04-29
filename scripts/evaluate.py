"""Dense-matrix evaluation script for MariHA continual learning runs.

For every per-task checkpoint discovered under
``experiments/checkpoints/{run_label}/{run_prefix}_task{k}/`` and for every
unique scene in the subject's curriculum, this script runs ``n_episodes``
of the greedy policy and writes the result into a 2D matrix:

    returns_matrix[run_index][scene_id] = mean return

Behavioral stats (clear rate, x-distance, score, deaths) are aggregated
in a parallel ``behavioral_matrix``. Per-scene and per-run metadata are
emitted alongside so downstream notebooks can derive whatever metric
they need (AP / BWT / forgetting / first-exposure-vs-final / etc.).

Usage::

    mariha-evaluate --subject sub-01 --agent sac --cl_method ewc \\
        --run_prefix 20260322_120000_seed0 --n_episodes 5

The ``--cl_method`` flag is optional; omit it to evaluate a vanilla
agent run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import mariha.rl  # noqa: F401 — registers all built-in agents
from mariha.benchmark.registry import get_agent_class
from mariha.curriculum.sequences import HumanSequence
from mariha.env.base import SCENARIOS_DIR
from mariha.env.scenario_gen import load_metadata
from mariha.eval.metrics import build_run_metadata, build_scene_metadata
from mariha.eval.runner import eval_on_scene, find_task_checkpoints
from mariha.utils.logging import EpochLogger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dense per-(checkpoint, scene) evaluation of a MariHA agent."
    )
    p.add_argument("--subject", required=True, help="Subject ID, e.g. sub-01.")
    p.add_argument("--agent", default="sac", help="Agent name (from registry).")
    p.add_argument(
        "--cl_method",
        default=None,
        help="Continual learning method composed during training "
        "(must match mariha-run-cl). Omit for vanilla runs.",
    )
    p.add_argument(
        "--run_prefix",
        required=True,
        help="Run prefix, e.g. '20260322_120000_seed0'.",
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
        help="Episodes per (checkpoint, scene) cell (default: 5).",
    )
    p.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Evaluate only the first N scenes (debug).",
    )
    p.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Evaluate only the first N checkpoints, ordered by run_index (debug).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_sizes", nargs="*", type=int, default=[])
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


def _build_eval_agent(agent_name: str, checkpoint_dir: Path, env, run_ids, args):
    """Instantiate an agent from the registry and load a checkpoint."""
    agent_cls = get_agent_class(agent_name)
    eval_logger = EpochLogger(
        output_dir="/tmp/mariha_eval_tmp",
        logger_output=["stdout"],
    )
    agent = agent_cls.from_args(args, env=env, logger=eval_logger, run_ids=run_ids)
    agent.load_checkpoint(checkpoint_dir)
    return agent


def _aggregate_behavioral(stats_list: list[dict]) -> dict:
    if not stats_list:
        return {}
    return {
        "clear_rate": float(np.mean([s.get("cleared", False) for s in stats_list])),
        "mean_x_traveled": float(np.mean([s.get("x_traveled", 0) for s in stats_list])),
        "mean_score_gained": float(np.mean([s.get("score_gained", 0) for s in stats_list])),
        "death_rate": float(np.mean([s.get("lives_lost", 0) for s in stats_list])),
    }


def main() -> None:
    parser = build_eval_parser()
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    agent_name = args.agent
    cl_name = args.cl_method
    run_label = f"{agent_name}_{cl_name}" if cl_name else agent_name
    checkpoint_base = experiment_dir / "checkpoints" / run_label

    task_checkpoints = find_task_checkpoints(checkpoint_base, args.run_prefix)
    if not task_checkpoints:
        logger.error(
            "No checkpoints found in %s matching prefix '%s'.",
            checkpoint_base, args.run_prefix,
        )
        sys.exit(1)

    sequence = HumanSequence(subject_id=args.subject)
    run_ids = sequence.run_ids
    scene_metadata = build_scene_metadata(sequence)
    run_metadata = build_run_metadata(sequence)

    eval_specs: dict = {}
    for spec in sequence:
        if spec.scene_id not in eval_specs:
            eval_specs[spec.scene_id] = spec

    scene_meta_emu = load_metadata(SCENARIOS_DIR)
    scenes_to_eval = sorted(eval_specs.keys())
    if args.max_scenes is not None:
        scenes_to_eval = scenes_to_eval[: args.max_scenes]

    checkpoint_indexes = sorted(task_checkpoints.keys())
    if args.max_checkpoints is not None:
        checkpoint_indexes = checkpoint_indexes[: args.max_checkpoints]

    logger.info(
        "Dense eval: %d checkpoints x %d scenes x %d episodes",
        len(checkpoint_indexes), len(scenes_to_eval), args.n_episodes,
    )

    from mariha.env.continual import make_scene_env

    first_scene = scenes_to_eval[0]
    dummy_env = make_scene_env(
        scene_id=first_scene,
        exit_point=scene_meta_emu[first_scene]["exit_point"],
        run_id=run_ids[0],
        run_ids=run_ids,
    )

    returns_matrix: dict[str, dict[str, float]] = {}
    behavioral_matrix: dict[str, dict[str, dict]] = {}

    try:
        for ci, run_index in enumerate(checkpoint_indexes):
            ckpt_path = task_checkpoints[run_index]
            logger.info(
                "[%d/%d] checkpoint run_index=%d (%s)",
                ci + 1, len(checkpoint_indexes), run_index, ckpt_path,
            )
            agent = _build_eval_agent(
                agent_name, ckpt_path, dummy_env, run_ids, args,
            )

            row_returns: dict[str, float] = {}
            row_behavioral: dict[str, dict] = {}
            for si, scene_id in enumerate(scenes_to_eval):
                spec = eval_specs[scene_id]
                mean_ret, stats = eval_on_scene(
                    agent, scene_id, spec, run_ids,
                    n_episodes=args.n_episodes,
                    scenarios_dir=SCENARIOS_DIR,
                )
                row_returns[scene_id] = float(mean_ret)
                row_behavioral[scene_id] = _aggregate_behavioral(stats)
                logger.info(
                    "    [%3d/%3d] %s  return=%.2f  clear=%.0f%%",
                    si + 1, len(scenes_to_eval), scene_id,
                    mean_ret,
                    row_behavioral[scene_id].get("clear_rate", 0) * 100,
                )

            returns_matrix[str(run_index)] = row_returns
            behavioral_matrix[str(run_index)] = row_behavioral
    finally:
        dummy_env.close()

    results = {
        "metadata": {
            "subject": args.subject,
            "agent": agent_name,
            "cl_method": cl_name,
            "run_label": run_label,
            "run_prefix": args.run_prefix,
            "n_episodes": args.n_episodes,
            "n_scenes_evaluated": len(scenes_to_eval),
            "n_checkpoints_evaluated": len(checkpoint_indexes),
            "n_runs_in_curriculum": len(run_ids),
        },
        "returns_matrix": returns_matrix,
        "behavioral_matrix": behavioral_matrix,
        "scene_metadata": scene_metadata,
        "run_metadata": {str(k): v for k, v in run_metadata.items()},
    }

    if args.output:
        out_path = Path(args.output)
    else:
        run_dir = experiment_dir / args.subject / run_label / args.run_prefix
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "eval_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
