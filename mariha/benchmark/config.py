"""Benchmark-level argument parsing and context construction.

``build_benchmark_parser()`` returns an argument parser containing only flags
that are agent-independent.  Agent-specific flags are added separately by
each agent's ``add_args()`` classmethod.

``build_benchmark_context()`` constructs the curriculum, environment, and
logger that every agent receives at construction time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from mariha.utils.running import get_readable_timestamp, str2bool


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    """Add the agent-agnostic flags shared by ``run_cl`` and ``run_single``.

    These are the flags that every entry point needs regardless of whether
    it runs a full curriculum or a single scene: agent identity, subject,
    seed, experiment dir, render config, curriculum filtering, and logger
    backends.
    """
    p.add_argument(
        "--agent",
        type=str,
        default="sac",
        help="Agent name (must be registered in mariha.rl). "
             "Run `mariha-run-cl --agent <name> --help` to see "
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
    p.add_argument(
        "--progress",
        type=str,
        default="live",
        choices=["live", "line", "off"],
        help="Terminal progress display. 'live' = persistent rich dashboard "
             "(default), 'line' = one enriched line per episode, "
             "'off' = legacy per-episode print only.",
    )


def build_benchmark_parser() -> argparse.ArgumentParser:
    """Return a parser containing only agent-agnostic benchmark flags.

    These flags are common to every agent and are always parsed first.
    Agent-specific flags (learning rate, network size, etc.) are added by
    the agent's ``add_args()`` classmethod.

    Returns:
        An ``ArgumentParser`` with benchmark-level flags only.
    """
    p = argparse.ArgumentParser(
        description="MariHA benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,  # help is added by the full parser in run_cl.py
    )
    _add_common_flags(p)

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

    This is the standard setup shared by every agent.  It is extracted
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
    from mariha.utils.progress import build_progress

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
    agent_name = getattr(args, "agent", None) or "sac"
    timestamp = get_readable_timestamp()
    experiment_dir = Path(args.experiment_dir)
    run_dir = str(
        experiment_dir / args.subject / agent_name / f"{timestamp}_seed{args.seed}"
    )
    logger = EpochLogger(
        output_dir=run_dir,
        logger_output=args.logger_output,
        config=vars(args),
        group_id=f"{args.subject}_{agent_name}",
    )

    # Progress display — attached to the logger so agents can pick it up
    # via `getattr(self.logger, "progress", None)`.
    progress = build_progress(getattr(args, "progress", "live"), fallback_log=logger.log)
    progress.init_meta(
        agent_name=agent_name,
        subject=args.subject,
        seed=args.seed,
        clip_total=len(sequence),
        total_steps=None,
    )
    logger.progress = progress

    return env, scene_ids, logger, sequence


def build_single_scene_parser() -> argparse.ArgumentParser:
    """Return a parser for ``mariha-run-single`` (single-scene debugging mode).

    Contains the common flags plus ``--scene_id`` (which scene to train on)
    and ``--total_steps`` (env-step budget).  Burn-in / buffer-mode /
    fixed-episode flags from ``build_benchmark_parser`` are intentionally
    omitted — they would be misleading no-ops in single-scene mode.

    Returns:
        An ``ArgumentParser`` with single-scene flags only.
    """
    p = argparse.ArgumentParser(
        description="MariHA single-scene runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,  # help is added by the full parser in run_single.py
    )
    _add_common_flags(p)
    p.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene ID to train on (e.g. 'w1l1s0'). "
             "If omitted, uses the first scene in the subject's curriculum.",
    )
    p.add_argument(
        "--total_steps",
        type=int,
        default=200_000,
        help="Total environment steps for single-scene training. "
             "Soft budget — the in-flight episode is allowed to finish.",
    )
    return p


def build_single_scene_context(args: argparse.Namespace) -> Tuple:
    """Construct a single-scene env, scene_ids, logger, and spec sequence.

    Mirrors ``build_benchmark_context`` but feeds the agent a
    ``StepBudgetCLEnv`` cycling forever over a single scene's episode specs,
    capped at ``args.total_steps`` env steps.

    Args:
        args: Parsed namespace from ``build_single_scene_parser`` (plus any
            agent-specific flags added downstream).

    Returns:
        A four-tuple ``(env, scene_ids, logger, sequence)`` matching the shape
        returned by ``build_benchmark_context`` so the same agent constructor
        path works in both modes.

    Raises:
        SystemExit: If the scene cannot be resolved or has no episodes.
    """
    import itertools

    from mariha.curriculum.loader import load_curriculum
    from mariha.env.continual import StepBudgetCLEnv
    from mariha.env.scenario_gen import load_metadata
    from mariha.env.base import SCENARIOS_DIR
    from mariha.utils.logging import EpochLogger
    from mariha.utils.progress import build_progress

    # Canonical scene ID ordering (alphabetical, consistent across runs).
    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())

    # Resolve scene_id: explicit arg or first scene in the subject's curriculum.
    all_specs = load_curriculum(
        subject_id=args.subject,
        require_existing_states=args.require_existing_states,
    )
    if not all_specs:
        print(f"ERROR: No episodes found for {args.subject}. Aborting.")
        sys.exit(1)

    scene_id = args.scene_id
    if scene_id is None:
        scene_id = all_specs[0].scene_id
        print(f"[run_single] No --scene_id given; using first curriculum scene: {scene_id}")

    if scene_id not in scene_meta:
        print(
            f"ERROR: Scene '{scene_id}' not found in scenario metadata. "
            f"Run `mariha-generate-scenarios` first."
        )
        sys.exit(1)

    scene_specs = [s for s in all_specs if s.scene_id == scene_id]
    if not scene_specs:
        print(
            f"ERROR: Scene '{scene_id}' has no episodes in {args.subject}'s curriculum."
        )
        sys.exit(1)

    # Cycle the specs forever; StepBudgetCLEnv stops at args.total_steps.
    spec_cycle = itertools.cycle(scene_specs)

    env = StepBudgetCLEnv(
        sequence=spec_cycle,
        scene_ids=scene_ids,
        max_steps=args.total_steps,
        render_mode=args.render_mode,
        render_speed=args.render_speed,
    )

    # Logger path: experiments/single/{scene_id}/{agent}/{timestamp}_seed{seed}
    agent_name = getattr(args, "agent", None) or "sac"
    timestamp = get_readable_timestamp()
    experiment_dir = Path(args.experiment_dir)
    run_dir = str(
        experiment_dir / "single" / scene_id / agent_name / f"{timestamp}_seed{args.seed}"
    )
    logger = EpochLogger(
        output_dir=run_dir,
        logger_output=args.logger_output,
        config=vars(args),
        group_id=f"single_{scene_id}_{agent_name}",
    )

    # In single-scene mode the cycled spec set has no fixed length; the
    # primary progress signal is the env-step budget instead.
    progress = build_progress(getattr(args, "progress", "live"), fallback_log=logger.log)
    progress.init_meta(
        agent_name=agent_name,
        subject=args.subject,
        seed=args.seed,
        clip_total=len(scene_specs),
        total_steps=args.total_steps,
    )
    logger.progress = progress

    return env, scene_ids, logger, scene_specs
