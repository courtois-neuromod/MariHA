#!/usr/bin/env python3
"""Replay model BK2 files and produce mp4 or gif, variables, state, and summary outputs.

Outputs are written next to each .bk2 file:
    <name>_recording.mp4  (default)
    <name>_recording.gif  (with --gif)
    <name>_variables.json
    <name>.state
    <name>_summary.json

Usage:
    python scripts/replay_model_bk2.py --bk2_dir <path> [-v] [--gif] [--gif-fps 24] [--gif-scale W]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import stable_retro as retro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mariha.env.base import GAME_NAME, STIMULI_PATH

_MARIO_SCENES_CODE = Path(__file__).resolve().parents[1] / "data" / "mario.scenes" / "code"
sys.path.insert(0, str(_MARIO_SCENES_CODE))
from utils import compute_clip_stats

FPS = 60.1  # approximate NES frame rate


def parse_bk2_name(stem: str) -> dict:
    """Extract BIDS-style entities from a model bk2 filename stem."""
    parts = {}
    for entity in stem.split("_"):
        if "-" in entity:
            key, _, val = entity.partition("-")
            parts[key] = val
    return parts


def replay_bk2(bk2_path: Path) -> tuple[list, list, bytes, list, list]:
    """Replay a bk2 and return (frames, infos, initial_state_bytes, all_keys, button_names)."""
    movie = retro.Movie(str(bk2_path))
    movie.step()  # advance to first frame to read initial state

    env = retro.make(
        GAME_NAME,
        state=retro.State.NONE,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )
    initial_state = movie.get_state()
    env.initial_state = initial_state
    env.reset()

    button_names: list = getattr(env.unwrapped, "buttons", []) or []
    n_buttons = len(button_names) if button_names else env.action_space.n
    frames: list[np.ndarray] = []
    infos: list[dict] = []
    all_keys: list[list] = []

    while movie.step():
        keys = [movie.get_key(i, 0) for i in range(n_buttons)]
        obs, _, _, _, _ = env.step(np.array(keys, dtype=np.uint8))
        frames.append(obs)
        info = env.data.lookup_all()
        info["x_pos"] = int(info.get("player_x_posHi", 0)) * 256 + int(info.get("player_x_posLo", 0))
        infos.append(info)
        all_keys.append(keys)

    env.close()
    movie.close()
    return frames, infos, initial_state, all_keys, button_names


def write_mp4(frames: list[np.ndarray], out_path: Path) -> None:
    """Write RGB frames to an mp4 via an intermediate AVI + ffmpeg remux."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_gif(frames: list[np.ndarray], out_path: Path, fps: int = 24, scale: int = 256) -> None:
    """Write RGB frames to a GIF via ffmpeg palette method."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = tmp.name
    palette_path = tmp_path + "_palette.png"
    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        vf_base = f"fps={fps},scale={scale}:-1:flags=lanczos"
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-vf", f"{vf_base},palettegen", palette_path],
            check=True, capture_output=True, stdin=subprocess.DEVNULL,
        )
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path, "-i", palette_path,
                "-filter_complex", f"[0:v]{vf_base}[x];[x][1:v]paletteuse",
                str(out_path),
            ],
            capture_output=True, stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, result.args,
                output=result.stdout, stderr=result.stderr,
            )
    finally:
        for p in (tmp_path, palette_path):
            if os.path.exists(p):
                os.unlink(p)


def build_variables(infos: list[dict], all_keys: list, button_names: list) -> dict:
    """Convert per-frame info dicts + button states to a dict of lists."""
    if not infos:
        return {}
    variables = {k: [float(d.get(k, 0)) for d in infos] for k in infos[0]}
    for i, btn in enumerate(button_names):
        if btn is not None:
            variables[f"button_{btn}"] = [int(row[i]) for row in all_keys]
    return variables


def _detect_outcome(infos: list[dict]) -> str:
    """Return 'death' or 'completed' by scanning per-frame game variables."""
    for i in range(1, len(infos)):
        ps_curr = infos[i].get("player_state", 0)
        ps_prev = infos[i - 1].get("player_state", 0)
        lives_curr = infos[i].get("lives", 0)
        lives_prev = infos[i - 1].get("lives", 0)
        if (ps_curr == 11 and ps_prev != 11) or lives_curr < lives_prev:
            return "death"
    return "completed"


def _build_scene_id(level: str, scene: str) -> str:
    """Reconstruct SceneID (e.g. 'w1l1s3') from level '1-1' and scene '3'."""
    try:
        world, level_num = level.split("-")
        return f"w{world}l{level_num}s{int(scene)}"
    except Exception:
        return f"level{level}s{scene}"


def process_bk2(bk2_path: Path, gif: bool = False, gif_fps: int = 24, gif_scale: int = 256) -> None:
    stem = bk2_path.stem
    video_path = bk2_path.parent / f"{stem}_recording.{'gif' if gif else 'mp4'}"
    state_path = bk2_path.parent / f"{stem}.state"
    vars_path = bk2_path.parent / f"{stem}_variables.json"
    summary_path = bk2_path.parent / f"{stem}_summary.json"

    if all(p.exists() for p in [video_path, state_path, vars_path, summary_path]):
        logging.info("Skipping (already exists): %s", bk2_path.name)
        return

    logging.info("Replaying: %s", bk2_path.name)
    frames, infos, initial_state, all_keys, button_names = replay_bk2(bk2_path)
    logging.info("  %d frames captured", len(frames))

    if gif:
        write_gif(frames, video_path, fps=gif_fps, scale=gif_scale)
    else:
        write_mp4(frames, video_path)

    with gzip.open(state_path, "wb") as fh:
        fh.write(initial_state)

    variables = build_variables(infos, all_keys, button_names)
    with open(vars_path, "w") as f:
        json.dump(variables, f)

    meta = parse_bk2_name(stem)
    clip_code = meta.get("clip", "")
    level = meta.get("level", "")
    scene = meta.get("scene", "")
    # clip_code encodes SSSRRBBNNNNNNN — run is digits [3:5]
    run = clip_code[3:5] if len(clip_code) >= 5 else None

    clip_stats = {}
    try:
        clip_stats = compute_clip_stats(variables)
    except Exception as e:
        logging.warning("compute_clip_stats failed: %s", e)

    summary = {
        "Subject": meta.get("sub"),
        "Session": meta.get("ses"),
        "Run": run,
        "Level": level,
        "SceneID": _build_scene_id(level, scene),
        "ClipCode": clip_code,
        "StartFrame": 0,
        "EndFrame": len(frames),
        "Duration": round(len(frames) / FPS, 3),
        "Outcome": _detect_outcome(infos),
        "Phase": None,
        "SourceBk2": bk2_path.name,
        "GameName": GAME_NAME,
        "Model": meta.get("model"),
        **clip_stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay model BK2 files.")
    parser.add_argument("--bk2_dir", required=True, help="Directory containing .bk2 files.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--gif", action="store_true", help="Output GIF instead of MP4.")
    parser.add_argument("--gif-fps", type=int, default=24, help="GIF frame rate (default: 24).")
    parser.add_argument("--gif-scale", type=int, default=256, help="GIF width in pixels (default: 256).")
    args = parser.parse_args()

    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
        format="%(levelname)s: %(message)s",
    )

    bk2_dir = Path(args.bk2_dir)
    bk2_files = sorted(bk2_dir.glob("*.bk2"))
    if not bk2_files:
        print(f"No .bk2 files found in {bk2_dir}")
        sys.exit(1)

    retro.data.Integrations.add_custom_path(str(STIMULI_PATH))
    print(f"Found {len(bk2_files)} bk2 files in {bk2_dir}")

    errors = []
    for i, bk2_path in enumerate(bk2_files, 1):
        print(f"[{i}/{len(bk2_files)}] {bk2_path.name}")
        try:
            process_bk2(bk2_path, gif=args.gif, gif_fps=args.gif_fps, gif_scale=args.gif_scale)
        except Exception as e:
            msg = f"ERROR {bk2_path.name}: {e}\n{traceback.format_exc()}"
            logging.error(msg)
            errors.append(msg)

    n_ok = len(bk2_files) - len(errors)
    print(f"\nDone. {n_ok}/{len(bk2_files)} files processed successfully.")
    if errors:
        err_log = bk2_dir / "replay_errors.log"
        err_log.write_text("\n".join(errors))
        print(f"Error log: {err_log}")


if __name__ == "__main__":
    main()
