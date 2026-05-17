#!/usr/bin/env python
"""Generate model BK2 replay files for one BIDS run after continual learning.

For a given human run (``run_id``) and a trained model (``run_prefix``), this
script loads the checkpoint saved right after the model trained on that run
(``task_idx == run_index``) and generates one BK2 replay file per episode
spec (clip) in that run.  The result is a 1:1 model BK2 for every human BK2,
enabling run-level human/model comparison.

Usage::

    python scripts/generate_model_bk2.py \\
        --subject sub-01 --agent dqn \\
        --run_prefix 2026_05_04__13_15_00_seed0 \\
        --run_id ses-001_run-02
"""

from __future__ import annotations

import argparse
import logging
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
        default="experiments",
        help="Root experiments directory (default: experiments).",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for BK2 files. "
            "Default: {experiment_dir}/{subject}/{run_label}/{run_prefix}/model_bk2/{run_id}/"
        ),
    )
    # Agent constructor flags — same set as evaluate.py
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_sizes", nargs="*", type=int, default=[])
    p.add_argument("--activation", default="tanh")
    p.add_argument("--use_layer_norm", action="store_true")
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--hide_task_id", action="store_true")
    return p


def _build_agent(agent_name: str, checkpoint_dir: Path, env, run_ids: list, args):
    agent_cls = get_agent_class(agent_name)
    eval_logger = EpochLogger(
        output_dir="/tmp/mariha_bk2_tmp",
        logger_output=["stdout"],
    )
    agent = agent_cls.from_args(args, env=env, logger=eval_logger, run_ids=run_ids)
    agent.load_checkpoint(checkpoint_dir)
    return agent


def _model_bk2_filename(spec, run_label: str) -> str:
    sub = spec.subject.replace("sub-", "")
    ses = spec.session.replace("ses-", "")
    scene_num = int(spec.scene_id.split("s")[-1])
    return (
        f"sub-{sub}_ses-{ses}_task-mario"
        f"_level-{spec.level}"
        f"_scene-{scene_num}"
        f"_clip-{spec.clip_code}"
        f"_model-{run_label}.bk2"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    agent_name = args.agent
    cl_name = args.cl_method
    run_label = f"{agent_name}_{cl_name}" if cl_name else agent_name
    checkpoint_base = experiment_dir / "checkpoints" / run_label

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
    # 2. Locate checkpoint                                                  #
    # ------------------------------------------------------------------ #
    checkpoint_dir = checkpoint_base / f"{args.run_prefix}_task{run_index}"
    if not checkpoint_dir.exists():
        logger.error("Checkpoint directory not found: %s", checkpoint_dir)
        sys.exit(1)

    logger.info("Checkpoint: %s", checkpoint_dir)

    # ------------------------------------------------------------------ #
    # 3. Determine output directory                                         #
    # ------------------------------------------------------------------ #
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = (
            experiment_dir
            / args.subject
            / run_label
            / args.run_prefix
            / "model_bk2"
            / args.run_id
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_dir)

    # ------------------------------------------------------------------ #
    # 4. Build dummy env → instantiate and load agent → close dummy env    #
    # stable-retro allows only one emulator per process; the dummy env     #
    # must be closed before the per-clip recording envs are opened.        #
    # ------------------------------------------------------------------ #
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
        agent = _build_agent(agent_name, checkpoint_dir, dummy_env, run_ids, args)
    finally:
        dummy_env.close()

    # ------------------------------------------------------------------ #
    # 5. Generate one BK2 per clip                                         #
    # ------------------------------------------------------------------ #
    n_generated = 0
    for i, spec in enumerate(run_specs):
        bk2_name = _model_bk2_filename(spec, run_label)
        final_path = out_dir / bk2_name

        if final_path.exists():
            logger.info(
                "[%d/%d] Already exists, skipping: %s",
                i + 1, len(run_specs), bk2_name,
            )
            n_generated += 1
            continue

        logger.info(
            "[%d/%d] clip=%s scene=%s",
            i + 1, len(run_specs), spec.clip_code, spec.scene_id,
        )

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
                obs, info = env.reset(episode_spec=spec)
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
            logger.info("  -> %s", bk2_name)
            n_generated += 1

    print(f"\nDone. {n_generated}/{len(run_specs)} BK2 files generated in:\n  {out_dir}")


if __name__ == "__main__":
    main()
