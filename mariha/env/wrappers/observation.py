"""Observation wrappers for the Mario environment.

Applies the standard Atari-style preprocessing pipeline used in COOM:

1. ``GrayscaleWrapper``  — convert RGB frames to single-channel grayscale.
2. ``ResizeWrapper``     — resize frames to 84 × 84 pixels.
3. ``FrameStackWrapper`` — stack the last N frames along the channel axis.
4. ``TaskIdWrapper``     — normalize pixels to [0,1]; expose task one-hot in info.

Each wrapper follows a minimal duck-typing interface compatible with both
``SceneEnv`` and ``ContinualLearningEnv``:  ``reset(**kwargs)``, ``step(action)``,
``render()``, ``close()``, ``observation_space``, ``action_space``.
"""

from __future__ import annotations

import collections
from typing import Any

import cv2
import numpy as np

from gymnasium import spaces

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_HEIGHT = 84
FRAME_WIDTH = 84
FRAME_STACK = 4


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------


class GrayscaleWrapper:
    """Convert RGB observations to grayscale.

    Args:
        env: The environment to wrap.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        h, w, _ = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )
        self.action_space = env.action_space

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._obs(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def _obs(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class ResizeWrapper:
    """Resize frames to ``(FRAME_HEIGHT, FRAME_WIDTH)`` pixels.

    Args:
        env: The environment to wrap.
        height: Target height in pixels. Default: 84.
        width: Target width in pixels. Default: 84.
    """

    def __init__(
        self,
        env: Any,
        height: int = FRAME_HEIGHT,
        width: int = FRAME_WIDTH,
    ) -> None:
        self._env = env
        _, _, c = env.observation_space.shape
        self._height = height
        self._width = width
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, c), dtype=np.uint8
        )
        self.action_space = env.action_space

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._obs(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def _obs(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized


# ---------------------------------------------------------------------------
# Frame stack
# ---------------------------------------------------------------------------


class FrameStackWrapper:
    """Stack the last ``n_frames`` observations along the channel axis.

    On ``reset()``, the buffer is filled with copies of the first frame.

    Args:
        env: The environment to wrap.
        n_frames: Number of frames to stack. Default: 4.
    """

    def __init__(self, env: Any, n_frames: int = FRAME_STACK) -> None:
        self._env = env
        self._n_frames = n_frames

        h, w, c = env.observation_space.shape
        self._buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=n_frames
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * n_frames),
            dtype=np.uint8,
        )
        self.action_space = env.action_space

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(**kwargs)
        for _ in range(self._n_frames):
            self._buffer.append(obs)
        return self._stack(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._buffer.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._buffer), axis=-1)


# ---------------------------------------------------------------------------
# Frame skip
# ---------------------------------------------------------------------------


class FrameSkipWrapper:
    """Repeat each action for ``n_skip`` steps and accumulate reward.

    Args:
        env: The environment to wrap.
        n_skip: Number of raw steps to repeat each action. Default: 4.
    """

    def __init__(self, env: Any, n_skip: int = 4) -> None:
        self._env = env
        self._n_skip = n_skip
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        return self._env.reset(**kwargs)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        for _ in range(self._n_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward / 10.0, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped


# ---------------------------------------------------------------------------
# Task ID
# ---------------------------------------------------------------------------


class TaskIdWrapper:
    """Normalize pixel observations and expose the task one-hot vector in info.

    The pixel observation ``(H, W, C)`` is normalized to ``[0, 1]`` float32
    and returned as-is (shape preserved). The one-hot task vector is placed
    in ``info['task_one_hot']`` so the agent can pass it separately to the
    actor/critic networks, matching the COOM two-input convention.

    Under the task=BIDS-run model, one-hot dimension = ``len(run_ids)`` and
    the active index is ``run_ids.index(run_id)``.

    Args:
        env: The environment to wrap (output of ``FrameStackWrapper``).
        run_id: The current run identifier (e.g. ``'ses-001_run-02'``).
        run_ids: Ordered list of all run IDs defining the one-hot index.
            Must be consistent across all calls (same ordering).
    """

    def __init__(
        self,
        env: Any,
        run_id: str,
        run_ids: list[str],
    ) -> None:
        self._env = env
        self._run_id = run_id
        self._run_ids = run_ids
        self._num_tasks = len(run_ids)

        if run_id not in run_ids:
            raise ValueError(
                f"run_id '{run_id}' not found in run_ids list."
            )
        self._task_idx = run_ids.index(run_id)

        # Normalized pixel observation — same shape as incoming frame stack.
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(h, w, c),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(**kwargs)
        info["task_one_hot"] = self._one_hot()
        return self._normalize(obs), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def set_run_id(self, run_id: str) -> None:
        """Update the active run ID (called by ``ContinualLearningEnv`` on task switch).

        Args:
            run_id: New run identifier.

        Raises:
            ValueError: If ``run_id`` is not in the known run list.
        """
        if run_id not in self._run_ids:
            raise ValueError(
                f"run_id '{run_id}' not found in run_ids list."
            )
        self._run_id = run_id
        self._task_idx = self._run_ids.index(run_id)

    def _normalize(self, frame_stack: np.ndarray) -> np.ndarray:
        return frame_stack.astype(np.float32) / 255.0

    def _one_hot(self) -> np.ndarray:
        vec = np.zeros(self._num_tasks, dtype=np.float32)
        vec[self._task_idx] = 1.0
        return vec
