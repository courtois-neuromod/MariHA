#!/usr/bin/env python
"""Generate model BK2 replay files for one BIDS session after continual learning.

For a given subject session and a trained model (``run_prefix``), this script
replays every clip of every run in that session. Each run is replayed with the
checkpoint saved right after the model finished training on that run
(``task_idx == run_index``), playing one greedy episode per clip.

Design notes
------------
* **Greedy policy.** Actions are sampled with ``deterministic=True``.
* **Checkpoint matched to the run.** Each run is replayed with the checkpoint
  saved immediately after the model trained on that run — so model and human
  are directly comparable on every run.
* **No human frame budget.** The episode is *not* truncated at the human's
  clip length; the model plays until it naturally clears the scene or dies.
  ``--max_episode_steps`` is only a large safety cap against infinite loops.
* **Hyperparameters from the run's config.json.** The agent is rebuilt from
  the training run's ``config.json`` so the network matches the checkpoint
  exactly; CLI agent flags are only a fallback when that file is missing.

Output mirrors the ``mario.scenes`` layout — one ``gamelogs.tar`` per session::

    {MARIHA_DATA_ROOT}/mario.scenes.{agent}-{cl_method|vanilla}/
        sub-XX/ses-XXX/gamelogs.tar   (members: gamelogs/..._clip-CCCC.bk2)

One invocation handles one whole session and is the *sole* writer of that
session's ``gamelogs.tar``. The SLURM array runs one task per session, so
different tasks only ever touch different files: there is never a concurrent
writer, no per-run intermediate archives, and no separate consolidation step.
The tar is written atomically (temp file + rename), so it is never observed
half-written even if the task is killed.

Usage::

    python scripts/generate_model_bk2.py \\
        --subject sub-01 --agent dqn --cl_method packnet \\
        --run_prefix 2026_05_04__13_15_00_seed0 \\
        --session ses-001 \\
        --experiment_dir $SCRATCH/MariHA/experiments
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mariha.rl  # noqa: F401 — registers all built-in agents
from mariha.benchmark.registry import get_agent_class
from mariha.curriculum.sequences import HumanSequence
from mariha.env.base import SCENARIOS_DIR
from mariha.env.continual import make_scene_env
from mariha.env.scenario_gen import load_metadata
from mariha.utils.logging import EpochLogger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

#: Safety cap on a model episode's length. Far longer than any human clip —
#: it only exists so a stuck policy cannot record forever.
DEFAULT_MAX_EPISODE_STEPS = 50_000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate model BK2 replay files for one BIDS session."
    )
    p.add_argument("--subject", required=True, help="Subject ID, e.g. sub-01.")
    p.add_argument("--agent", default="sac", help="Agent name (from registry).")
    p.add_argument(
        "--cl_method",
        default=None,
        help="Continual learning method composed during training. Omit for vanilla runs.",
    )
    p.add_argument(
        "--run_prefix",
        required=True,
        help="Run prefix, e.g. '2026_05_04__13_15_00_seed0'.",
    )
    p.add_argument(
        "--session",
        required=True,
        help="BIDS session to generate BK2s for, e.g. 'ses-001'.",
    )
    p.add_argument(
        "--experiment_dir",
        default=os.environ.get("MARIHA_EXPERIMENT_DIR", "experiments"),
        help="Root experiments directory (default: $MARIHA_EXPERIMENT_DIR or 'experiments').",
    )
    p.add_argument(
        "--bids_root",
        default=None,
        help=(
            "Output BIDS dataset root. Default: "
            "{MARIHA_DATA_ROOT}/mario.scenes.{agent}-{cl_method|vanilla}/"
        ),
    )
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Safety cap on model episode length (not a human-matched budget).",
    )
    # Agent constructor flags — fallback only, used when config.json is absent.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_sizes", nargs="*", type=int, default=[])
    p.add_argument("--activation", default="tanh")
    p.add_argument("--use_layer_norm", action="store_true")
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--hide_task_id", action="store_true")
    return p


def _resolve_bids_root(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Path:
    """Resolve the output dataset root: mario.scenes.{agent}-{cl|vanilla}."""
    if args.bids_root:
        return Path(args.bids_root)
    data_root = os.environ.get("MARIHA_DATA_ROOT")
    if not data_root:
        parser.error("Pass --bids_root or set MARIHA_DATA_ROOT.")
    combo = (
        f"{args.agent}-{args.cl_method}" if args.cl_method else f"{args.agent}-vanilla"
    )
    return Path(data_root) / f"mario.scenes.{combo}"


def _load_train_config(
    experiment_dir: Path, subject: str, run_label: str, run_prefix: str,
    fallback: argparse.Namespace,
) -> argparse.Namespace:
    """Return the training run's args, read from its ``config.json``.

    Falls back to the CLI ``fallback`` namespace (with a warning) when the
    config file cannot be found — e.g. for ad-hoc runs.
    """
    config_path = (
        experiment_dir / subject / run_label / run_prefix / "config.json"
    )
    if not config_path.exists():
        logger.warning(
            "config.json not found at %s — rebuilding the agent from CLI "
            "flags instead. The network may not match the checkpoint.",
            config_path,
        )
        return fallback
    with open(config_path) as f:
        cfg = json.load(f)
    logger.info("Loaded training config: %s", config_path)
    return argparse.Namespace(**cfg)


def _build_agent(agent_name: str, checkpoint_dir: Path, env, run_ids: list,
                  train_args: argparse.Namespace):
    agent_cls = get_agent_class(agent_name)
    eval_logger = EpochLogger(
        output_dir="/tmp/mariha_bk2_tmp",
        logger_output=["stdout"],
    )
    agent = agent_cls.from_args(train_args, env=env, logger=eval_logger, run_ids=run_ids)
    agent.load_checkpoint(checkpoint_dir)
    return agent


def _bk2_filename(spec) -> str:
    """Return the BIDS-style ``.bk2`` filename for a clip's model replay.

    Pattern: ``{sub}_{ses}_task-mario_run-NN_level-L_scene-S_clip-CCCC.bk2``.
    """
    run = spec.run_id.split("run-")[-1]
    scene_num = int(spec.scene_id.split("s")[-1])
    return (
        f"{spec.subject}_{spec.session}_task-mario_run-{run}"
        f"_level-{spec.level}_scene-{scene_num}_clip-{spec.clip_code}.bk2"
    )


def _session_archive_path(bids_root: Path, subject: str, session: str) -> Path:
    """Return a session's ``gamelogs`` tar.

    Mirrors the ``mario.scenes`` layout: ``{bids_root}/{sub}/{ses}/gamelogs.tar``.
    """
    return bids_root / subject / session / "gamelogs.tar"


def _pack_gamelogs(staging: Path, archive: Path) -> None:
    """Pack a session's staged ``gamelogs/`` folder into its tar archive.

    Written to a PID-suffixed temp file and atomically ``os.replace``-d into
    place, so ``gamelogs.tar`` is never observed half-written even if the task
    is killed mid-write. An uncompressed ``.tar`` keeps consistency with the
    ``mario.scenes`` source dataset.
    """
    tmp = archive.with_name(f"{archive.name}.tmp.{os.getpid()}")
    try:
        with tarfile.open(tmp, "w") as tar:
            tar.add(staging, arcname="gamelogs")
        os.replace(tmp, archive)
    finally:
        tmp.unlink(missing_ok=True)


def _generate_run(
    *, run_id: str, run_specs: list, run_ids: list, checkpoint_dir: Path,
    agent_name: str, train_args: argparse.Namespace, scene_meta: dict,
    staging: Path, max_episode_steps: int,
) -> int:
    """Replay every clip of one run into ``staging``. Returns the clip count.

    stable-retro allows only one emulator per process, so the dummy env used
    to build the agent is closed before any per-clip recording env is opened,
    and each recording env is closed before the next.
    """
    first_spec = run_specs[0]
    first_exit = scene_meta[first_spec.scene_id]["exit_point"]
    dummy_env = make_scene_env(
        scene_id=first_spec.scene_id,
        exit_point=first_exit,
        run_id=run_id,
        run_ids=run_ids,
    )
    try:
        agent = _build_agent(
            agent_name, checkpoint_dir, dummy_env, run_ids, train_args
        )
    finally:
        dummy_env.close()

    n_generated = 0
    for i, spec in enumerate(run_specs):
        logger.info(
            "  [%d/%d] clip=%s scene=%s",
            i + 1, len(run_specs), spec.clip_code, spec.scene_id,
        )

        # Lift the human-derived frame budget: the model plays to natural
        # termination (scene cleared or death), capped only for safety.
        play_spec = dataclasses.replace(spec, max_steps=max_episode_steps)

        with tempfile.TemporaryDirectory() as rec_dir:
            rec_path = Path(rec_dir)
            exit_point = scene_meta[spec.scene_id]["exit_point"]

            env = make_scene_env(
                scene_id=spec.scene_id,
                exit_point=exit_point,
                run_id=run_id,
                run_ids=run_ids,
                record_dir=rec_path,
            )
            try:
                obs, info = env.reset(episode_spec=play_spec)
                one_hot = info["task_one_hot"]
                done = False
                while not done:
                    action = agent.get_action(obs, one_hot, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
            finally:
                env.unwrapped.stop_record()
                env.close()

            bk2_files = list(rec_path.glob("*.bk2"))
            if not bk2_files:
                logger.warning("  No BK2 produced for clip %s — skipping.", spec.clip_code)
                continue
            if len(bk2_files) > 1:
                logger.warning(
                    "  Multiple BK2 files found for clip %s; using first.", spec.clip_code
                )

            shutil.move(str(bk2_files[0]), str(staging / _bk2_filename(spec)))
            n_generated += 1

    return n_generated


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    bids_root = _resolve_bids_root(args, parser)
    experiment_dir = Path(args.experiment_dir)
    run_label = (
        f"{args.agent}_{args.cl_method}" if args.cl_method else args.agent
    )

    # ------------------------------------------------------------------ #
    # 1. Load curriculum → the runs that make up this session              #
    # ------------------------------------------------------------------ #
    sequence = HumanSequence(subject_id=args.subject)
    run_ids = sequence.run_ids
    session_run_ids = [r for r in run_ids if r.split("_run-")[0] == args.session]

    if not session_run_ids:
        logger.error(
            "Session '%s' not found in curriculum for subject '%s'.",
            args.session, args.subject,
        )
        sys.exit(1)

    session_archive = _session_archive_path(bids_root, args.subject, args.session)
    if session_archive.exists():
        logger.info(
            "Session %s already archived (%s) — nothing to do.",
            args.session, session_archive,
        )
        return

    logger.info(
        "Session %s → %d runs; output dataset %s",
        args.session, len(session_run_ids), bids_root,
    )

    # ------------------------------------------------------------------ #
    # 2. Replay every clip of every run into one staging folder            #
    # ------------------------------------------------------------------ #
    train_args = _load_train_config(
        experiment_dir, args.subject, run_label, args.run_prefix, fallback=args,
    )
    scene_meta = load_metadata(SCENARIOS_DIR)

    session_archive.parent.mkdir(parents=True, exist_ok=True)
    n_generated = 0
    n_clips = 0
    with tempfile.TemporaryDirectory() as staging_root:
        staging = Path(staging_root) / "gamelogs"
        staging.mkdir()

        for run_id in session_run_ids:
            run_index = run_ids.index(run_id)
            run_specs = [s for s in sequence if s.run_id == run_id]
            n_clips += len(run_specs)

            # Checkpoint saved right after the model trained on this run.
            checkpoint_dir = (
                experiment_dir / "checkpoints" / run_label / args.subject
                / f"{args.run_prefix}_task{run_index}"
            )
            if not checkpoint_dir.exists():
                logger.error(
                    "Run %s: checkpoint not found (%s) — skipping this run.",
                    run_id, checkpoint_dir,
                )
                continue

            logger.info(
                "Run %s (task %d, %d clips) — checkpoint %s",
                run_id, run_index, len(run_specs), checkpoint_dir.name,
            )
            n = _generate_run(
                run_id=run_id,
                run_specs=run_specs,
                run_ids=run_ids,
                checkpoint_dir=checkpoint_dir,
                agent_name=args.agent,
                train_args=train_args,
                scene_meta=scene_meta,
                staging=staging,
                max_episode_steps=args.max_episode_steps,
            )
            logger.info("Run %s: %d/%d clips generated.", run_id, n, len(run_specs))
            n_generated += n

        # ----------------------------------------------------------- #
        # 3. Pack the whole session into gamelogs.tar in one atomic    #
        #    write — this task is the only writer of that file.        #
        # ----------------------------------------------------------- #
        if n_generated == 0:
            logger.warning(
                "Session %s: no BK2 generated — gamelogs.tar not written.",
                args.session,
            )
        else:
            _pack_gamelogs(staging, session_archive)
            logger.info(
                "Session %s archived → %s (%d clips)",
                args.session, session_archive, n_generated,
            )

    print(
        f"\nDone. {n_generated}/{n_clips} BK2 files generated for session "
        f"'{args.session}'."
    )


if __name__ == "__main__":
    main()
