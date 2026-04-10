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


def play_render_episode(
    actor_fn,
    scene_id: str,
    exit_point: int,
    scene_ids: list[str],
    spec,
    stimuli_path: Path = STIMULI_PATH,
    scenarios_dir: Path = SCENARIOS_DIR,
    render_speed: float = 1.0,
) -> None:
    """Create a human-render env, play one greedy episode with the given policy, close it.

    This is the shared implementation used by ``ContinualLearningEnv.render_checkpoint()``
    (which both ``run_single.py`` and ``run_cl.py`` route through).  Callers are
    responsible for closing any existing emulator instance before calling this
    function (stable-retro allows only one emulator per process).

    Args:
        actor_fn: Callable ``(obs, one_hot) -> int`` — the current policy.
        scene_id: Scene to render.
        exit_point: X-coordinate goal for the scene.
        scene_ids: Full ordered list of scene IDs (for one-hot encoding).
        spec: ``EpisodeSpec`` to load as the starting state.
        stimuli_path: Override for stimuli directory.
        scenarios_dir: Override for scenario files directory.
        render_speed: Speed multiplier. 1 = 60 fps, 0.5 = 30 fps,
            10 = 600 fps (best effort).
    """
    import time

    render_env = make_scene_env(
        scene_id=scene_id,
        exit_point=exit_point,
        scene_ids=scene_ids,
        render_mode="human",
        stimuli_path=stimuli_path,
        scenarios_dir=scenarios_dir,
    )
    obs, info = render_env.reset(episode_spec=spec)
    one_hot = info["task_one_hot"]
    target_dt = 1.0 / (render_speed * 60.0)
    done = False
    while not done:
        t0 = time.monotonic()
        action = actor_fn(obs, one_hot)
        obs, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        elapsed = time.monotonic() - t0
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    render_env.close()


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
        render_speed: Speed multiplier for rendering (1 = 60 fps).
        stimuli_path: Override for the stimuli directory.
        scenarios_dir: Override for the scenario files directory.
    """

    def __init__(
        self,
        sequence: Any,
        scene_ids: list[str],
        render_mode: str | None = None,
        render_speed: float = 1.0,
        stimuli_path: Path = STIMULI_PATH,
        scenarios_dir: Path = SCENARIOS_DIR,
    ) -> None:
        self._sequence_iter: Iterator = iter(sequence)
        self._scene_ids = scene_ids
        self._render_mode = render_mode
        self._render_speed = render_speed
        self._stimuli_path = stimuli_path
        self._scenarios_dir = scenarios_dir

        # Loaded metadata for exit points.
        self._scene_metadata = load_metadata(scenarios_dir)

        # Active environment and tracking state.
        self._env: TaskIdWrapper | None = None
        self._current_scene_id: str | None = None
        self._next_spec: Any = None  # prefetched from sequence
        self._current_spec: Any = None  # spec of the episode currently being played
        self._episode_count: int = 0
        self._done: bool = False
        self._current_session: str | None = None

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
        self._current_spec = spec
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
        info["session"] = spec.session
        info["session_switch"] = (
            self._current_session is not None
            and spec.session != self._current_session
        )
        self._current_session = spec.session
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

    def render_checkpoint(self, actor_fn) -> None:
        """Play one greedy episode with human rendering, then restore the training env.

        Temporarily closes the inner scene env so the emulator can be reused
        for rendering (stable-retro allows only one emulator per process).
        The curriculum sequence state is not affected.

        Args:
            actor_fn: Callable ``(obs, one_hot) -> int`` — the current policy.
        """
        if self._current_scene_id is None or self._current_spec is None:
            return
        if self._env is not None:
            self._env.close()
        meta = self._scene_metadata[self._current_scene_id]
        play_render_episode(
            actor_fn=actor_fn,
            scene_id=self._current_scene_id,
            exit_point=meta["exit_point"],
            scene_ids=self._scene_ids,
            spec=self._current_spec,
            stimuli_path=self._stimuli_path,
            scenarios_dir=self._scenarios_dir,
            render_speed=self._render_speed,
        )
        self._build_env(self._current_scene_id)

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


class StepBudgetCLEnv(ContinualLearningEnv):
    """Continual learning env that stops after a fixed environment-step budget.

    Used by ``mariha-run-single`` to feed any registered agent a single-scene
    curriculum that terminates after ``max_steps`` env steps. The wrapped
    sequence is typically ``itertools.cycle(scene_specs)`` so that the agent
    can keep resetting until the budget is exhausted.

    **Soft-budget semantics**: agents only check ``env.is_done`` at episode
    boundaries (see ``SAC.run``, ``PPO.run``, ``DQN.run``), so the actual
    step count can overshoot ``max_steps`` by up to one episode length.
    This matches the previous ``run_single.py`` behavior, which also let the
    in-flight episode complete before exiting.

    Args:
        sequence: Iterable of ``EpisodeSpec`` objects (typically a cycled list
            of specs from a single scene).
        scene_ids: Ordered list of all valid scene IDs.
        max_steps: Total environment-step budget. Once reached, ``is_done``
            flips to ``True`` at the next episode boundary.
        render_mode: Render mode for ``MarioEnv``.
        render_speed: Speed multiplier for rendering (1 = 60 fps).
        stimuli_path: Override for the stimuli directory.
        scenarios_dir: Override for the scenario files directory.
    """

    def __init__(
        self,
        sequence: Any,
        scene_ids: list[str],
        max_steps: int,
        render_mode: str | None = None,
        render_speed: float = 1.0,
        stimuli_path: Path = STIMULI_PATH,
        scenarios_dir: Path = SCENARIOS_DIR,
    ) -> None:
        super().__init__(
            sequence=sequence,
            scene_ids=scene_ids,
            render_mode=render_mode,
            render_speed=render_speed,
            stimuli_path=stimuli_path,
            scenarios_dir=scenarios_dir,
        )
        self._max_steps = int(max_steps)
        self._step_count = 0

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self._done = True
        return obs, reward, terminated, truncated, info
