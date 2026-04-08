"""Exact episode replay from a saved trajectory file.

Loads a ``.npz`` trajectory produced by ``run_single.py``, re-steps the
environment with the exact sequence of actions that occurred during training,
captures raw NES RGB frames, and writes an MP4 video.

Because the same emulator ``.state`` file is loaded and the same actions are
replayed in order, the resulting video is byte-for-byte identical to what
happened during training.

Usage::

    python scripts/replay_episode.py \\
        --trajectory experiments/single/w1l1s0/<run>/trajectories/episode_00001.npz \\
        --output /tmp/ep1.mp4

Or via the entry point (if configured in pyproject.toml)::

    mariha-replay --trajectory <path> --output <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mariha.curriculum.episode import EpisodeSpec
from mariha.env.base import SCENARIOS_DIR, STIMULI_PATH
from mariha.env.continual import make_scene_env
from mariha.env.scenario_gen import load_metadata
from mariha.utils.video import write_episode_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a training episode exactly from a saved trajectory .npz."
    )
    parser.add_argument(
        "--trajectory", type=str, required=True,
        help="Path to an episode_NNNNN.npz trajectory file.",
    )
    parser.add_argument(
        "--output", type=str, default="replay.mp4",
        help="Output MP4 file path. Default: replay.mp4",
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Frames per second for the output video. Default: 60",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load trajectory
    # ------------------------------------------------------------------ #
    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        print(f"ERROR: trajectory file not found: {traj_path}")
        sys.exit(1)

    data = np.load(traj_path, allow_pickle=True)
    actions = data["actions"].astype(np.int32)
    state_file = Path(str(data["state_file"]))
    scene_id = str(data["scene_id"])
    episode_index = int(data["episode_index"])
    ep_return = float(data["ep_return"])

    print(
        f"Trajectory: episode {episode_index}  |  scene {scene_id}  |  "
        f"{len(actions)} steps  |  return {ep_return:.1f}"
    )

    if not state_file.exists():
        print(f"ERROR: state file not found: {state_file}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Build environment
    # ------------------------------------------------------------------ #
    scene_meta = load_metadata(SCENARIOS_DIR)
    if scene_id not in scene_meta:
        print(f"ERROR: scene '{scene_id}' not found in scenario metadata.")
        sys.exit(1)

    exit_point = scene_meta[scene_id]["exit_point"]
    scene_ids = sorted(scene_meta.keys())

    env = make_scene_env(
        scene_id=scene_id,
        exit_point=exit_point,
        scene_ids=scene_ids,
        render_mode="rgb_array",
        stimuli_path=STIMULI_PATH,
        scenarios_dir=SCENARIOS_DIR,
    )

    # Minimal EpisodeSpec — only state_file and max_steps are used by reset().
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

    # ------------------------------------------------------------------ #
    # Replay
    # ------------------------------------------------------------------ #
    env.reset(episode_spec=spec)
    frames: list[np.ndarray] = []

    # Capture the initial frame before the first step.
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for i, action in enumerate(actions):
        _, _, terminated, truncated, _ = env.step(int(action))
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    env.close()

    if not frames:
        print("ERROR: no frames captured during replay.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write video
    # ------------------------------------------------------------------ #
    write_episode_video(frames, args.output, fps=args.fps)
    print(f"Video written: {args.output}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
