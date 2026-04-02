"""Video writing utility for MariHA episode visualization.

Provides a thin wrapper around ``cv2.VideoWriter`` for saving lists of RGB
frames to MP4 files.  Used by the replay and visualization scripts.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_episode_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: int = 60,
) -> None:
    """Write a sequence of RGB frames to an MP4 file.

    Args:
        frames: List of ``(H, W, 3)`` uint8 RGB arrays, one per environment
            step.  All frames must have the same shape.
        output_path: Destination file path (e.g. ``"replay.mp4"``).  Parent
            directories are created automatically if they do not exist.
        fps: Frames per second for the output video.  Defaults to ``60`` to
            match the NES emulator frame rate.

    Raises:
        ValueError: If ``frames`` is empty.
        RuntimeError: If ``cv2.VideoWriter`` fails to open the output file.
    """
    if not frames:
        raise ValueError("frames must not be empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open output file: {output_path}"
        )

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
