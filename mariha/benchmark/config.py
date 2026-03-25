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
        add_help=False,  # help is added by the full parser in run_benchmark.py
    )

    p.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        help="Algorithm name (must be registered in mariha.rl). "
             "Run `mariha-run --help` after selecting an algorithm to see "
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
        A three-tuple ``(env, scene_ids, logger)`` where:

        - ``env`` is a ``ContinualLearningEnv`` instance.
        - ``scene_ids`` is the canonical ordered list of all scene IDs.
        - ``logger`` is an ``EpochLogger`` configured for this run.

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

    # Canonical scene ID ordering (alphabetical, consistent across runs)
    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())

    # Environment
    env = ContinualLearningEnv(
        sequence=sequence,
        scene_ids=scene_ids,
        render_mode=args.render_mode,
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

    return env, scene_ids, logger
