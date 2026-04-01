"""Continual learning environment: sequences episodes across tasks.

``ContinualLearningEnv`` is the top-level environment used during training.
It wraps a fully-stacked ``SceneEnv`` (with all observation and action
wrappers applied) and drives the episode sequence provided by the curriculum.

On each ``reset()`` call:
- The next ``EpisodeSpec`` is popped from the sequence.
- If the scene has changed since the last episode, ``task_switch=True`` is
  set in the returned ``info`` dict so that CL methods can update their
  importance weights, prune masks, etc.
- The ``TaskIdWrapper`` is updated to reflect the new scene ID.

The episode sequence is provided by a ``BaseSequence`` (see
``mariha.curriculum.sequences``).  The default is ``HumanSequence``, which
replays human clips in ``clip_code`` order.

Args:
    sequence: A ``BaseSequence`` instance that yields ``EpisodeSpec`` objects.
    scene_ids: Ordered list of all valid scene IDs (defines the one-hot
        task vector index).  Use ``mariha.env.scenario_gen.load_metadata``
        to obtain the full list.
    render_mode: Passed through to ``MarioEnv``.
    seed: Optional global RNG seed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from gymnasium import spaces

from mariha.env.base import STIMULI_PATH, SCENARIOS_DIR
from mariha.env.scene import SceneEnv
from mariha.env.wrappers.action import ActionWrapper
from mariha.env.wrappers.observation import (
    FrameStackWrapper,
    GrayscaleWrapper,
    ResizeWrapper,
    TaskIdWrapper,
)
from mariha.env.scenario_gen import load_metadata

logger = logging.getLogger(__name__)


def make_scene_env(
    scene_id: str,
    exit_point: int,
    scene_ids: list[str],
    render_mode: str | None = None,
    stimuli_path: Path = STIMULI_PATH,
    scenarios_dir: Path = SCENARIOS_DIR,
) -> TaskIdWrapper:
    """Build and return a fully-wrapped scene environment.

    Applies the full observation pipeline (grayscale → resize → frame-stack
    → task-ID) and the discrete action wrapper.

    Args:
        scene_id: Scene identifier (e.g. ``'w1l1s0'``).
        exit_point: X-coordinate at which the scene is considered cleared.
        scene_ids: Ordered list of all scene IDs for the one-hot encoding.
        render_mode: Render mode for ``MarioEnv``.
        stimuli_path: Override for the stimuli directory.
        scenarios_dir: Override for the scenario files directory.

    Returns:
        A ``TaskIdWrapper``-wrapped env ready for the SAC training loop.
    """
    env = SceneEnv(
        scene_id=scene_id,
        exit_point=exit_point,
        render_mode=render_mode,
        stimuli_path=stimuli_path,
        scenarios_dir=scenarios_dir,
    )
    env = ActionWrapper(env)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    env = FrameStackWrapper(env)
    env = TaskIdWrapper(env, scene_id=scene_id, scene_ids=scene_ids)
    return env


class ContinualLearningEnv:
    """Sequences episodes from a curriculum across multiple scenes.

    The environment creates a single underlying ``SceneEnv`` stack and
    reconfigures it at each task boundary.  This avoids the overhead of
    creating a new retro environment per task.

    Note: A new ``SceneEnv`` is created when the scene changes (because the
    retro scenario file is scene-specific).  Within a scene, the same env
    instance is reused across episodes.

    Args:
        sequence: A ``BaseSequence`` (or any iterable of ``EpisodeSpec``).
        scene_ids: Ordered list of all valid scene IDs.
        render_mode: Render mode for ``MarioEnv``.
        stimuli_path: Override for the stimuli directory.
        scenarios_dir: Override for the scenario files directory.
    """

    def __init__(
        self,
        sequence: Any,
        scene_ids: list[str],
        render_mode: str | None = None,
        stimuli_path: Path = STIMULI_PATH,
        scenarios_dir: Path = SCENARIOS_DIR,
    ) -> None:
        self._sequence_iter: Iterator = iter(sequence)
        self._scene_ids = scene_ids
        self._render_mode = render_mode
        self._stimuli_path = stimuli_path
        self._scenarios_dir = scenarios_dir

        # Loaded metadata for exit points.
        self._scene_metadata = load_metadata(scenarios_dir)

        # Active environment and tracking state.
        self._env: TaskIdWrapper | None = None
        self._current_scene_id: str | None = None
        self._next_spec: Any = None  # prefetched from sequence
        self._episode_count: int = 0
        self._done: bool = False

        # Prefetch the first spec so we can set up observation/action spaces.
        self._prefetch_next()
        if self._next_spec is not None:
            self._build_env(self._next_spec.scene_id)

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Space:
        """Observation space (set by the first scene's env)."""
        if self._env is None:
            raise RuntimeError("ContinualLearningEnv has no loaded environment.")
        return self._env.observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Discrete action space (same for all scenes)."""
        if self._env is None:
            raise RuntimeError("ContinualLearningEnv has no loaded environment.")
        return self._env.action_space

    def reset(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset to the next episode in the sequence.

        Returns:
            ``(observation, info)`` where ``info`` contains:
            - ``task_switch`` (bool): True if the scene ID changed.
            - ``scene_id`` (str): Current scene.
            - ``episode_index`` (int): Global episode counter.
            - All fields from ``SceneEnv._add_episode_info()``.

        Raises:
            StopIteration: When the sequence is exhausted.
        """
        if self._done or self._next_spec is None:
            raise StopIteration("Curriculum sequence is exhausted.")

        spec = self._next_spec
        task_switch = spec.scene_id != self._current_scene_id

        # Rebuild the env if the task changed.
        if task_switch:
            if self._env is not None:
                self._env.close()
            self._build_env(spec.scene_id)
            self._current_scene_id = spec.scene_id
            logger.info("Task switch → %s", spec.scene_id)
        else:
            # Same scene: just update the task ID wrapper (no-op here,
            # but makes the interface consistent).
            self._env.set_scene_id(spec.scene_id)  # type: ignore[union-attr]

        obs, info = self._env.reset(episode_spec=spec, seed=seed)
        info["task_switch"] = task_switch
        info["episode_index"] = self._episode_count

        self._episode_count += 1
        self._prefetch_next()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the active scene environment.

        Args:
            action: Discrete action index.

        Returns:
            Standard gymnasium tuple.
        """
        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        obs, reward, terminated, truncated, info = self._env.step(action)
        info["episode_index"] = self._episode_count - 1
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current frame."""
        if self._env is None:
            return None
        return self._env.render()

    def close(self) -> None:
        """Close the active environment."""
        if self._env is not None:
            self._env.close()

    @property
    def is_done(self) -> bool:
        """True when the curriculum sequence is exhausted."""
        return self._done

    @property
    def episode_stats(self):
        """Episode stats from the most recently completed episode."""
        if self._env is None:
            return None
        # Traverse wrapper chain to reach SceneEnv.
        inner = self._env
        while hasattr(inner, "_env"):
            inner = inner._env
        return getattr(inner, "episode_stats", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prefetch_next(self) -> None:
        """Advance the sequence iterator by one step."""
        try:
            self._next_spec = next(self._sequence_iter)
        except StopIteration:
            self._next_spec = None
            self._done = True

    def _build_env(self, scene_id: str) -> None:
        """Create a fresh wrapped env for ``scene_id``.

        Args:
            scene_id: Scene identifier to build the env for.
        """
        meta = self._scene_metadata.get(scene_id)
        if meta is None:
            raise ValueError(
                f"No metadata found for scene '{scene_id}'. "
                "Run `mariha-generate-scenarios` first."
            )
        self._env = make_scene_env(
            scene_id=scene_id,
            exit_point=meta["exit_point"],
            scene_ids=self._scene_ids,
            render_mode=self._render_mode,
            stimuli_path=self._stimuli_path,
            scenarios_dir=self._scenarios_dir,
        )
