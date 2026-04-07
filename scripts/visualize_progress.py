"""Progress visualization: watch the agent improve over training.

Replays selected episodes from a ``run_single.py`` output directory and saves
one MP4 per episode, allowing you to compare the agent's behaviour at
different points in training.

Two modes are supported:

``trajectory`` (default, exact)
    Replays the actual actions logged during training.  Requires
    ``--save_trajectories`` to have been enabled (the default).

``weights`` (approximate)
    Loads interval weight checkpoints and runs the agent fresh on the scene.
    Requires ``--checkpoint_every`` to have been set during training.

Usage — trajectory mode::

    python scripts/visualize_progress.py \\
        --run_dir experiments/single/w1l1s0/<run> \\
        --every 10 \\
        --output_dir /tmp/progress

Usage — weights mode::

    python scripts/visualize_progress.py \\
        --run_dir experiments/single/w1l1s0/<run> \\
        --mode weights \\
        --episodes 1,5,10 \\
        --subject sub-01 \\
        --scene_id w1l1s0 \\
        --output_dir /tmp/progress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf

from mariha.curriculum.episode import EpisodeSpec
from mariha.curriculum.loader import load_curriculum
from mariha.env.base import SCENARIOS_DIR, STIMULI_PATH
from mariha.env.continual import make_scene_env
from mariha.env.scenario_gen import load_metadata
from mariha.rl import models
from mariha.utils.running import get_activation_from_str
from mariha.utils.video import write_episode_video


def _replay_trajectory(traj_path: Path, env, output_path: Path, fps: int) -> dict:
    """Replay one trajectory file and write an MP4. Returns summary dict."""
    data = np.load(traj_path, allow_pickle=True)
    actions = data["actions"].astype(np.int32)
    state_file = Path(str(data["state_file"]))
    scene_id = str(data["scene_id"])
    episode_index = int(data["episode_index"])
    ep_return = float(data["ep_return"])

    if not state_file.exists():
        return {"episode": episode_index, "skipped": True, "reason": "state file missing"}

    spec = EpisodeSpec(
        state_file=state_file,
        max_steps=len(actions),
        scene_id=scene_id,
        clip_code="00000000000000",
        subject="replay",
        session="ses-000",
        outcome="unknown",
        phase="replay",
        level=scene_id[:4],
    )

    env.reset(episode_spec=spec)
    frames: list[np.ndarray] = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for action in actions:
        _, _, terminated, truncated, _ = env.step(int(action))
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    if frames:
        write_episode_video(frames, output_path, fps=fps)

    return {
        "episode": episode_index,
        "scene_id": scene_id,
        "ep_return": ep_return,
        "steps": len(actions),
        "frames": len(frames),
        "output": str(output_path),
        "skipped": False,
    }


def _run_weights(
    ckpt_path: Path, spec: EpisodeSpec, env, actor, one_hot: np.ndarray,
    output_path: Path, fps: int, step_label: str,
) -> dict:
    """Run agent with loaded weights for one episode and write MP4."""
    if not (ckpt_path / "actor.index").exists():
        return {"label": step_label, "skipped": True, "reason": "weights missing"}

    actor.load_weights(str(ckpt_path / "actor"))

    obs, _ = env.reset(episode_spec=spec)
    frames: list[np.ndarray] = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    ep_return = 0.0
    done = False
    while not done:
        logits = actor(
            tf.expand_dims(tf.convert_to_tensor(obs), 0),
            tf.expand_dims(tf.convert_to_tensor(one_hot), 0),
        )
        action = int(tf.argmax(logits, axis=-1).numpy()[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        done = terminated or truncated
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    if frames:
        write_episode_video(frames, output_path, fps=fps)

    return {
        "label": step_label,
        "ep_return": ep_return,
        "frames": len(frames),
        "output": str(output_path),
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize agent progress over a single-scene training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to a run_single.py output directory.",
    )
    parser.add_argument(
        "--mode", type=str, default="trajectory", choices=["trajectory", "weights"],
        help="trajectory: exact replay from saved action logs. "
             "weights: approximate replay from interval weight checkpoints.",
    )
    parser.add_argument(
        "--episodes", type=str, default=None,
        help="Comma-separated episode indices to visualize (trajectory mode), "
             "e.g. '1,50,100'. Mutually exclusive with --every.",
    )
    parser.add_argument(
        "--every", type=int, default=None,
        help="Visualize every N episodes / checkpoints. "
             "Mutually exclusive with --episodes.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output MP4s. Defaults to <repo>/replay/visualize_progress/.",
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Frames per second for output videos.",
    )
    # Weights mode only
    parser.add_argument(
        "--subject", type=str, default="sub-01",
        help="(weights mode) Subject ID to load an episode spec from.",
    )
    parser.add_argument(
        "--scene_id", type=str, default=None,
        help="(weights mode) Scene to run the agent on.",
    )
    # Architecture flags for weights mode
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "relu", "elu", "lrelu"])
    parser.add_argument("--use_layer_norm", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--hide_task_id", action="store_true", default=False)
    args = parser.parse_args()

    if args.episodes and args.every:
        print("ERROR: --episodes and --every are mutually exclusive.")
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    # ------------------------------------------------------------------ #
    # Shared: scene metadata
    # ------------------------------------------------------------------ #
    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())

    # ------------------------------------------------------------------ #
    # Trajectory mode
    # ------------------------------------------------------------------ #
    if args.mode == "trajectory":
        traj_dir = run_dir / "trajectories"
        if not traj_dir.exists():
            print(
                f"ERROR: no trajectories/ directory found in {run_dir}.\n"
                "Re-run with --save_trajectories True (the default)."
            )
            sys.exit(1)

        all_trajs = sorted(traj_dir.glob("episode_*.npz"))
        if not all_trajs:
            print(f"ERROR: no trajectory files found in {traj_dir}.")
            sys.exit(1)

        # Select which trajectories to visualize.
        if args.episodes:
            wanted = {int(x) for x in args.episodes.split(",")}
            selected = [
                t for t in all_trajs
                if int(t.stem.split("_")[1]) in wanted
            ]
        elif args.every:
            selected = all_trajs[:: args.every]
        else:
            selected = all_trajs  # all episodes

        if not selected:
            print("ERROR: no matching trajectories found for the given selection.")
            sys.exit(1)

        # Infer scene_id from first trajectory to build env once.
        first_data = np.load(selected[0], allow_pickle=True)
        scene_id = str(first_data["scene_id"])
        exit_point = scene_meta[scene_id]["exit_point"]

        if output_dir is None:
            output_dir = REPO_ROOT / "replay" / "visualize_progress" / scene_id
        output_dir.mkdir(parents=True, exist_ok=True)

        env = make_scene_env(
            scene_id=scene_id,
            exit_point=exit_point,
            scene_ids=scene_ids,
            render_mode="rgb_array",
            stimuli_path=STIMULI_PATH,
            scenarios_dir=SCENARIOS_DIR,
        )

        print(f"Replaying {len(selected)} episode(s) from {traj_dir} …")
        for traj_path in selected:
            ep_idx = int(traj_path.stem.split("_")[1])
            out = output_dir / f"episode_{ep_idx:05d}.mp4"
            result = _replay_trajectory(traj_path, env, out, args.fps)
            if result.get("skipped"):
                print(f"  [skip] episode {ep_idx}: {result.get('reason')}")
            else:
                print(
                    f"  episode {result['episode']:>5d}  |  "
                    f"return {result['ep_return']:>7.1f}  |  "
                    f"{result['frames']} frames  →  {result['output']}"
                )

        env.close()

    # ------------------------------------------------------------------ #
    # Weights mode
    # ------------------------------------------------------------------ #
    else:
        ckpt_root = run_dir / "checkpoints"
        if not ckpt_root.exists():
            print(
                f"ERROR: no checkpoints/ directory found in {run_dir}.\n"
                "Re-run with --checkpoint_every <N> to save interval checkpoints."
            )
            sys.exit(1)

        if not args.scene_id:
            print("ERROR: --scene_id is required in weights mode.")
            sys.exit(1)
        if args.scene_id not in scene_meta:
            print(f"ERROR: scene '{args.scene_id}' not found in scenario metadata.")
            sys.exit(1)

        all_step_dirs = sorted(
            ckpt_root.glob("step_*"),
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not all_step_dirs:
            print(f"ERROR: no step_* checkpoint directories found in {ckpt_root}.")
            sys.exit(1)

        if args.episodes:
            wanted_steps = {int(x) for x in args.episodes.split(",")}
            selected_dirs = [
                d for d in all_step_dirs
                if int(d.name.split("_")[1]) in wanted_steps
            ]
        elif args.every:
            selected_dirs = all_step_dirs[:: args.every]
        else:
            selected_dirs = all_step_dirs

        if not selected_dirs:
            print("ERROR: no matching checkpoints for the given selection.")
            sys.exit(1)

        if output_dir is None:
            output_dir = REPO_ROOT / "replay" / "visualize_progress" / args.scene_id
        output_dir.mkdir(parents=True, exist_ok=True)

        exit_point = scene_meta[args.scene_id]["exit_point"]
        specs = load_curriculum(subject_id=args.subject, require_existing_states=True)
        scene_specs = [s for s in specs if s.scene_id == args.scene_id]
        if not scene_specs:
            print(
                f"ERROR: no episode specs for scene '{args.scene_id}' "
                f"under subject '{args.subject}'."
            )
            sys.exit(1)
        spec = scene_specs[0]

        env = make_scene_env(
            scene_id=args.scene_id,
            exit_point=exit_point,
            scene_ids=scene_ids,
            render_mode="rgb_array",
            stimuli_path=STIMULI_PATH,
            scenarios_dir=SCENARIOS_DIR,
        )

        task_idx = scene_ids.index(args.scene_id)
        one_hot = np.zeros(len(scene_ids), dtype=np.float32)
        one_hot[task_idx] = 1.0

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
        # Warm-up build.
        dummy_obs = np.zeros(env.observation_space.shape, dtype=np.float32)
        actor(
            tf.expand_dims(tf.convert_to_tensor(dummy_obs), 0),
            tf.expand_dims(tf.convert_to_tensor(one_hot), 0),
        )

        print(f"Running agent on {len(selected_dirs)} checkpoint(s) …")
        for ckpt_dir in selected_dirs:
            step_num = int(ckpt_dir.name.split("_")[1])
            out = output_dir / f"step_{step_num:07d}.mp4"
            result = _run_weights(
                ckpt_dir, spec, env, actor, one_hot, out, args.fps,
                step_label=f"step_{step_num}",
            )
            if result.get("skipped"):
                print(f"  [skip] {ckpt_dir.name}: {result.get('reason')}")
            else:
                print(
                    f"  step {step_num:>7d}  |  "
                    f"return {result['ep_return']:>7.1f}  |  "
                    f"{result['frames']} frames  →  {result['output']}"
                )

        env.close()

    print(f"\nDone. Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
