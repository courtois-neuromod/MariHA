"""Benchmark-level argument parsing and context construction.

``build_benchmark_parser()`` returns an argument parser containing only flags
that are algorithm-independent.  Algorithm-specific flags are added separately
by each algorithm's ``add_args()`` classmethod.

``build_benchmark_context()`` constructs the curriculum, environment, and
logger that every algorithm receives at construction time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from mariha.utils.running import get_readable_timestamp, str2bool


def build_benchmark_parser() -> argparse.ArgumentParser:
    """Return a parser containing only algorithm-agnostic benchmark flags.

    These flags are common to every algorithm and are always parsed first.
    Algorithm-specific flags (learning rate, network size, etc.) are added by
    the algorithm's ``add_args()`` classmethod.

    Returns:
        An ``ArgumentParser`` with benchmark-level flags only.
    """
    p = argparse.ArgumentParser(
        description="MariHA benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,  # help is added by the full parser in run_cl.py
    )

    p.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        help="Algorithm name (must be registered in mariha.rl). "
             "Run `mariha-run-cl --algorithm <name> --help` to see "
             "its specific flags.",
    )
    p.add_argument(
        "--subject",
        type=str,
        default="sub-01",
        help="Subject ID for the HumanSequence curriculum.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed.",
    )
    p.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Root directory for checkpoints and logs.",
    )
    p.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Render mode for MarioEnv (None or 'human').",
    )
    p.add_argument(
        "--render_speed",
        type=float,
        default=1.0,
        help="Render speed multiplier. 1 = 60 fps, 0.5 = 30 fps, "
             "10 = 600 fps (best effort).",
    )
    p.add_argument(
        "--require_existing_states",
        type=str2bool,
        default=True,
        help="Skip curriculum clips whose .state file is missing.",
    )
    p.add_argument(
        "--logger_output",
        type=str,
        nargs="+",
        default=["tsv", "tensorboard"],
        help="Logging backends: tsv, tensorboard, wandb.",
    )

    # Burn-in
    p.add_argument(
        "--burn_in_steps",
        type=int,
        default=5000,
        help="Environment steps on burn-in scene before curriculum starts. "
             "Set to 0 to disable burn-in.",
    )
    p.add_argument(
        "--burn_in_scene",
        type=str,
        default="w1l1s0",
        help="Scene ID for the burn-in phase.",
    )
    p.add_argument(
        "--post_burn_in_update_after",
        type=int,
        default=0,
        help="Value of update_after after burn-in completes. "
             "0 means updates start immediately once the curriculum begins.",
    )

    # Per-scene buffer mode
    p.add_argument(
        "--buffer_mode",
        type=str,
        default="single",
        choices=["single", "per_scene"],
        help="Replay buffer strategy: 'single' (one global buffer) or "
             "'per_scene' (per-scene buffers with session-boundary flush).",
    )
    p.add_argument(
        "--per_scene_capacity",
        type=int,
        default=1000,
        help="Max transitions per scene buffer (only with --buffer_mode per_scene).",
    )
    p.add_argument(
        "--flush_on",
        type=str,
        default="session",
        choices=["session", "level", "never"],
        help="When to flush per-scene buffers.",
    )
    p.add_argument(
        "--cl_hook_min_transitions",
        type=int,
        default=500,
        help="Minimum transitions in current scene buffer before firing CL hooks.",
    )

    # Episode budget overrides
    p.add_argument(
        "--fixed_episode_steps",
        type=int,
        default=None,
        help="Override max_steps for ALL episodes to this fixed value. "
             "Mutually exclusive with --episode_step_multiplier.",
    )
    p.add_argument(
        "--episode_step_multiplier",
        type=float,
        default=None,
        help="Scale each episode's human-derived max_steps by this factor. "
             "Mutually exclusive with --fixed_episode_steps.",
    )
    return p


def build_benchmark_context(args: argparse.Namespace) -> Tuple:
    """Construct the curriculum, environment, and logger from parsed args.

    This is the standard setup shared by every algorithm.  It is extracted
    here so ``run_benchmark.py`` and any shim scripts can reuse it without
    duplication.

    Args:
        args: Parsed namespace — must contain at minimum the flags defined by
            ``build_benchmark_parser()``.

    Returns:
        A four-tuple ``(env, scene_ids, logger, sequence)`` where:

        - ``env`` is a ``ContinualLearningEnv`` instance.
        - ``scene_ids`` is the canonical ordered list of all scene IDs.
        - ``logger`` is an ``EpochLogger`` configured for this run.
        - ``sequence`` is the curriculum ``BaseSequence`` (for burn-in spec
          extraction and other pre-training uses).

    Raises:
        SystemExit: If the curriculum for ``args.subject`` is empty.
    """
    from mariha.curriculum.sequences import HumanSequence
    from mariha.env.continual import ContinualLearningEnv
    from mariha.env.scenario_gen import load_metadata
    from mariha.env.base import SCENARIOS_DIR
    from mariha.utils.logging import EpochLogger

    # Curriculum
    sequence = HumanSequence(
        subject_id=args.subject,
        require_existing_states=args.require_existing_states,
    )
    if len(sequence) == 0:
        print(f"ERROR: No episodes found for {args.subject}. Aborting.")
        sys.exit(1)

    # Episode budget overrides
    fixed_steps = getattr(args, "fixed_episode_steps", None)
    multiplier = getattr(args, "episode_step_multiplier", None)
    if fixed_steps is not None and multiplier is not None:
        print("ERROR: --fixed_episode_steps and --episode_step_multiplier "
              "are mutually exclusive.")
        sys.exit(1)
    if fixed_steps is not None or multiplier is not None:
        from mariha.curriculum.sequences import BudgetOverrideSequence
        sequence = BudgetOverrideSequence(
            sequence, fixed_steps=fixed_steps, multiplier=multiplier,
        )

    # Canonical scene ID ordering (alphabetical, consistent across runs)
    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())

    # Environment
    env = ContinualLearningEnv(
        sequence=sequence,
        scene_ids=scene_ids,
        render_mode=args.render_mode,
        render_speed=args.render_speed,
    )

    # Logger
    algorithm_name = getattr(args, "algorithm", None) or "sac"
    timestamp = get_readable_timestamp()
    experiment_dir = Path(args.experiment_dir)
    run_dir = str(
        experiment_dir / args.subject / algorithm_name / f"{timestamp}_seed{args.seed}"
    )
    logger = EpochLogger(
        output_dir=run_dir,
        logger_output=args.logger_output,
        config=vars(args),
        group_id=f"{args.subject}_{algorithm_name}",
    )

    return env, scene_ids, logger, sequence
