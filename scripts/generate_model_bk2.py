#!/usr/bin/env python
"""Generate model BK2 replay files for one BIDS run after continual learning.

For a given human run (``run_id``) and a trained model (``run_prefix``), this
script loads the checkpoint saved right after the model trained on that run
(``task_idx == run_index``) and plays one greedy episode per clip in that run.

Design notes
------------
* **Greedy policy.** Actions are sampled with ``deterministic=True``.
* **Checkpoint matched to the clip.** The checkpoint used is the one saved
  immediately after the model finished training on the clip's own
  sub/ses/run — so model and human are directly comparable on that run.
* **No human frame budget.** The episode is *not* truncated at the human's
  clip length; the model plays until it naturally clears the scene or dies.
  ``--max_episode_steps`` is only a large safety cap against infinite loops.
* **Hyperparameters from the run's config.json.** The agent is rebuilt from
  the training run's ``config.json`` so the network matches the checkpoint
  exactly; CLI agent flags are only a fallback when that file is missing.

Output is a minimal-BIDS dataset, one per agent/CL-method combo::

    {MARIHA_DATA_ROOT}/mario.scenes.{agent}-{cl_method|vanilla}/
        sub-XX/ses-XXX/func/sub-XX_ses-XXX_task-mario_run-NN_..._clip-CCCC.bk2

Usage::

    python scripts/generate_model_bk2.py \\
        --subject sub-01 --agent dqn --cl_method packnet \\
        --run_prefix 2026_05_04__13_15_00_seed0 \\
        --run_id ses-001_run-02 \\
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
        description="Generate model BK2 replay files for one BIDS run."
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
        "--run_id",
        required=True,
        help="BIDS run to generate BK2s for, e.g. 'ses-001_run-02'.",
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


def _bids_bk2_path(bids_root: Path, spec) -> Path:
    """Return the minimal-BIDS path for a clip's model BK2.

    Layout: ``{bids_root}/{sub}/{ses}/func/{sub}_{ses}_task-mario_run-NN
    _level-L_scene-S_clip-CCCC.bk2``.
    """
    sub = spec.subject               # 'sub-01'
    ses = spec.session               # 'ses-002'
    run = spec.run_id.split("run-")[-1]
    scene_num = int(spec.scene_id.split("s")[-1])
    fname = (
        f"{sub}_{ses}_task-mario_run-{run}"
        f"_level-{spec.level}_scene-{scene_num}_clip-{spec.clip_code}.bk2"
    )
    return bids_root / sub / ses / "func" / fname


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    agent_name = args.agent
    cl_name = args.cl_method
    run_label = f"{agent_name}_{cl_name}" if cl_name else agent_name

    # Resolve the output BIDS dataset root: mario.scenes.{agent}-{cl|vanilla}.
    combo = f"{agent_name}-{cl_name}" if cl_name else f"{agent_name}-vanilla"
    if args.bids_root:
        bids_root = Path(args.bids_root)
    else:
        data_root = os.environ.get("MARIHA_DATA_ROOT")
        if not data_root:
            parser.error("Pass --bids_root or set MARIHA_DATA_ROOT.")
        bids_root = Path(data_root) / f"mario.scenes.{combo}"

    # ------------------------------------------------------------------ #
    # 1. Load curriculum → validate run_id, compute run_index              #
    # ------------------------------------------------------------------ #
    sequence = HumanSequence(subject_id=args.subject)
    run_ids = sequence.run_ids

    if args.run_id not in run_ids:
        logger.error(
            "run_id '%s' not found in curriculum for subject '%s'.",
            args.run_id, args.subject,
        )
        sys.exit(1)

    run_index = run_ids.index(args.run_id)
    run_specs = [s for s in sequence if s.run_id == args.run_id]

    if not run_specs:
        logger.error("No episode specs found for run_id '%s'.", args.run_id)
        sys.exit(1)

    logger.info(
        "Run '%s' → run_index=%d, %d clips", args.run_id, run_index, len(run_specs)
    )

    # ------------------------------------------------------------------ #
    # 2. Locate checkpoint (subject-namespaced, matched to this run)       #
    # ------------------------------------------------------------------ #
    checkpoint_dir = (
        experiment_dir
        / "checkpoints"
        / run_label
        / args.subject
        / f"{args.run_prefix}_task{run_index}"
    )
    if not checkpoint_dir.exists():
        logger.error("Checkpoint directory not found: %s", checkpoint_dir)
        sys.exit(1)

    logger.info("Checkpoint: %s", checkpoint_dir)

    # ------------------------------------------------------------------ #
    # 3. Output directory (minimal-BIDS dataset for this combo)            #
    # ------------------------------------------------------------------ #
    logger.info("Output dataset: %s", bids_root)

    # ------------------------------------------------------------------ #
    # 4. Rebuild the agent from the training run's config, load checkpoint #
    # stable-retro allows only one emulator per process; the dummy env     #
    # must be closed before the per-clip recording envs are opened.        #
    # ------------------------------------------------------------------ #
    train_args = _load_train_config(
        experiment_dir, args.subject, run_label, args.run_prefix, fallback=args,
    )

    scene_meta = load_metadata(SCENARIOS_DIR)
    first_spec = run_specs[0]
    first_exit = scene_meta[first_spec.scene_id]["exit_point"]

    dummy_env = make_scene_env(
        scene_id=first_spec.scene_id,
        exit_point=first_exit,
        run_id=args.run_id,
        run_ids=run_ids,
    )
    try:
        agent = _build_agent(agent_name, checkpoint_dir, dummy_env, run_ids, train_args)
    finally:
        dummy_env.close()

    # ------------------------------------------------------------------ #
    # 5. Generate one BK2 per clip — greedy policy, no human frame budget  #
    # ------------------------------------------------------------------ #
    n_generated = 0
    for i, spec in enumerate(run_specs):
        final_path = _bids_bk2_path(bids_root, spec)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        if final_path.exists():
            logger.info(
                "[%d/%d] Already exists, skipping: %s",
                i + 1, len(run_specs), final_path.name,
            )
            n_generated += 1
            continue

        logger.info(
            "[%d/%d] clip=%s scene=%s",
            i + 1, len(run_specs), spec.clip_code, spec.scene_id,
        )

        # Lift the human-derived frame budget: the model plays to natural
        # termination (scene cleared or death), capped only for safety.
        play_spec = dataclasses.replace(spec, max_steps=args.max_episode_steps)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            exit_point = scene_meta[spec.scene_id]["exit_point"]

            env = make_scene_env(
                scene_id=spec.scene_id,
                exit_point=exit_point,
                run_id=args.run_id,
                run_ids=run_ids,
                record_dir=tmpdir_path,
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

            bk2_files = list(tmpdir_path.glob("*.bk2"))
            if not bk2_files:
                logger.warning("No BK2 produced for clip %s — skipping.", spec.clip_code)
                continue
            if len(bk2_files) > 1:
                logger.warning(
                    "Multiple BK2 files found for clip %s; using first.", spec.clip_code
                )

            shutil.move(str(bk2_files[0]), str(final_path))
            logger.info("  -> %s", final_path.name)
            n_generated += 1

    print(f"\nDone. {n_generated}/{len(run_specs)} BK2 files generated in:\n  {bids_root}")


if __name__ == "__main__":
    main()
