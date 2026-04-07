"""Weights-based episode visualization for single-scene runs.

Loads saved actor weights produced by ``run_single.py --save_weights`` (or
``--checkpoint_every``), runs the agent greedily on one episode of the
requested scene, captures raw NES RGB frames, and writes an MP4 video.

Unlike ``replay_episode.py``, this is **not** an exact replay — the trained
policy is evaluated fresh and may choose different actions than it did during
training (the policy is stochastic by nature).  Use this script to evaluate
the final trained agent on any scene, including scenes not seen during
training.

Usage::

    python scripts/visualize_episode.py \\
        --subject sub-01 \\
        --scene_id w1l1s0 \\
        --checkpoint_dir experiments/single/w1l1s0/<run>/checkpoints/final \\
        --output /tmp/viz.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf

from mariha.curriculum.loader import load_curriculum
from mariha.env.base import SCENARIOS_DIR, STIMULI_PATH
from mariha.env.continual import make_scene_env
from mariha.env.scenario_gen import load_metadata
from mariha.rl import models
from mariha.utils.running import get_activation_from_str
from mariha.utils.video import write_episode_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the trained agent on a scene and record an MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subject", type=str, default="sub-01",
        help="Subject ID used to load an episode spec for the scene.",
    )
    parser.add_argument(
        "--scene_id", type=str, required=True,
        help="Scene to visualize (e.g. w1l1s0).",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Directory containing saved actor weights (actor.index / actor.data-*).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output MP4 file path. Default: <repo>/replay/visualize_episode/<scene_id>.mp4",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed.",
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Frames per second for the output video.",
    )
    # Architecture flags — must match the run_single.py defaults so that
    # weight loading succeeds.
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "relu", "elu", "lrelu"])
    parser.add_argument("--use_layer_norm", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--hide_task_id", action="store_true", default=False)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    output_path = args.output or str(REPO_ROOT / "replay" / "visualize_episode" / f"{args.scene_id}.mp4")

    ckpt_dir = Path(args.checkpoint_dir)
    if not (ckpt_dir / "actor.index").exists():
        print(f"ERROR: no actor weights found in {ckpt_dir}")
        print("Run with --save_weights (or --checkpoint_every) to produce checkpoints.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Scene metadata + episode spec
    # ------------------------------------------------------------------ #
    scene_meta = load_metadata(SCENARIOS_DIR)
    if args.scene_id not in scene_meta:
        print(f"ERROR: scene '{args.scene_id}' not found in scenario metadata.")
        sys.exit(1)

    scene_ids = sorted(scene_meta.keys())
    exit_point = scene_meta[args.scene_id]["exit_point"]

    specs = load_curriculum(subject_id=args.subject, require_existing_states=True)
    scene_specs = [s for s in specs if s.scene_id == args.scene_id]
    if not scene_specs:
        print(
            f"ERROR: no episode specs found for scene '{args.scene_id}' "
            f"under subject '{args.subject}'."
        )
        sys.exit(1)

    spec = scene_specs[0]

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    env = make_scene_env(
        scene_id=args.scene_id,
        exit_point=exit_point,
        scene_ids=scene_ids,
        render_mode="rgb_array",
        stimuli_path=STIMULI_PATH,
        scenarios_dir=SCENARIOS_DIR,
    )

    # ------------------------------------------------------------------ #
    # Reconstruct actor and load weights
    # ------------------------------------------------------------------ #
    activation = get_activation_from_str(args.activation)
    actor = models.MlpActor(
        state_space=env.observation_space,
        action_space=env.action_space,
        num_tasks=len(scene_ids),
        hidden_sizes=tuple(args.hidden_sizes),
        activation=activation,
        use_layer_norm=args.use_layer_norm,
        num_heads=args.num_heads,
        hide_task_id=args.hide_task_id,
    )

    task_idx = scene_ids.index(args.scene_id)
    one_hot = np.zeros(len(scene_ids), dtype=np.float32)
    one_hot[task_idx] = 1.0

    # Warm up the model so weights can be loaded (TF lazy build).
    dummy_obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    actor(
        tf.expand_dims(tf.convert_to_tensor(dummy_obs), 0),
        tf.expand_dims(tf.convert_to_tensor(one_hot), 0),
    )
    actor.load_weights(str(ckpt_dir / "actor"))
    print(f"Loaded weights from {ckpt_dir}")

    # ------------------------------------------------------------------ #
    # Run one episode greedily
    # ------------------------------------------------------------------ #
    obs, _ = env.reset(episode_spec=spec)
    frames: list[np.ndarray] = []

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    done = False
    while not done:
        logits = actor(
            tf.expand_dims(tf.convert_to_tensor(obs), 0),
            tf.expand_dims(tf.convert_to_tensor(one_hot), 0),
        )
        action = int(tf.argmax(logits, axis=-1).numpy()[0])
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        if frame is not None:
            frames.append(frame)

    env.close()

    if not frames:
        print("ERROR: no frames captured.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write video
    # ------------------------------------------------------------------ #
    write_episode_video(frames, output_path, fps=args.fps)
    print(f"Video written: {output_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
