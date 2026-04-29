"""Continual learning environment: sequences episodes across tasks.

``ContinualLearningEnv`` is the top-level environment used during training.
It wraps a fully-stacked ``SceneEnv`` (with all observation and action
wrappers applied) and drives the episode sequence provided by the curriculum.

Vocabulary (see ``docs/glossary.rst``): one BIDS fMRI run drives one
continual-learning task; one ``EpisodeSpec`` = one clip = one human attempt
at one scene; the scene is the Mario sub-region (e.g. ``w1l1s0``).

Every ``reset()`` advances exactly one clip — the clip is the atomic unit
of the curriculum.  The ``*_switch`` flags in ``info`` are conditions on
that transition (did the run / scene / session differ from the previous
clip's?):

- ``info["task_switch"]`` is ``True`` when the BIDS run changes (i.e. the
  agent has entered a new task). CL methods consume this to consolidate
  importance weights, prune masks, episodic memory, etc.
- ``info["scene_switch"]`` is ``True`` when the scene changes.  Gates the
  emulator rebuild (the retro scenario file is per-scene); also drives a
  diagnostic log line.  No agent hook fires on this.
- The ``TaskIdWrapper`` is updated to reflect the new run ID (the one-hot
  is task-level, i.e. per-run).
- The underlying ``SceneEnv`` is rebuilt whenever the scene changes (the
  retro scenario file is scene-specific), regardless of task boundaries.

The episode sequence is provided by a ``BaseSequence`` (see
``mariha.curriculum.sequences``).  The default is ``HumanSequence``, which
replays human clips in ``clip_code`` order.
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
    FrameSkipWrapper,
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
    run_id: str,
    run_ids: list[str],
    render_mode: str | None = None,
    stimuli_path: Path = STIMULI_PATH,
    scenarios_dir: Path = SCENARIOS_DIR,
) -> TaskIdWrapper:
    """Build and return a fully-wrapped scene environment.

    Applies the full observation pipeline (grayscale → resize → frame-stack
    → task-ID) and the discrete action wrapper. The scene identifies the
    physical emulator configuration; the run identifies the task for one-hot
    conditioning.

    Args:
        scene_id: Scene identifier (e.g. ``'w1l1s0'``). Drives the emulator.
        exit_point: X-coordinate at which the scene is considered cleared.
        run_id: Current task identifier (e.g. ``'ses-001_run-02'``).
        run_ids: Ordered list of all run IDs defining the task one-hot index.
        render_mode: Render mode for ``MarioEnv``.
        stimuli_path: Override for the stimuli directory.
        scenarios_dir: Override for the scenario files directory.

    Returns:
        A ``TaskIdWrapper``-wrapped env ready for the training loop.
    """
    env = SceneEnv(
        scene_id=scene_id,
        exit_point=exit_point,
        render_mode=render_mode,
        stimuli_path=stimuli_path,
        scenarios_dir=scenarios_dir,
    )
    env = ActionWrapper(env)
    env = FrameSkipWrapper(env, n_skip=4)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    env = FrameStackWrapper(env)
    env = TaskIdWrapper(env, run_id=run_id, run_ids=run_ids)
    return env


def play_render_episode(
    actor_fn,
    scene_id: str,
    exit_point: int,
    run_id: str,
    run_ids: list[str],
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
        run_id: Current task (run) identifier for the one-hot.
        run_ids: Full ordered list of run IDs (for one-hot encoding).
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
        run_id=run_id,
        run_ids=run_ids,
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

    Progress tracking is optional: if a ``TrainingProgress`` instance is
    passed via ``progress``, the env emits ``on_reset`` / ``on_episode_end``
    events automatically so individual agents don't need to know about
    the display layer at all.  ``on_episode_end`` is fired lazily from the
    *next* ``reset()`` call (or from ``close()`` for the final episode),
    which lets the agent's ``logger.store({...})`` run in between and
    surface agent-specific scalars (``buffer_fill_pct``, ``epsilon``, ...)
    to the display via ``EpochLogger.store``.

    Args:
        sequence: A ``BaseSequence`` (or any iterable of ``EpisodeSpec``).
        scene_ids: Ordered list of scene IDs the emulator knows about
            (from ``load_metadata``) — used for scene metadata lookups.
        run_ids: Ordered list of all run IDs in the subject's curriculum.
            Defines the task one-hot dimension and indexing.
        render_mode: Render mode for ``MarioEnv``.
        render_speed: Speed multiplier for rendering (1 = 60 fps).
        stimuli_path: Override for the stimuli directory.
        scenarios_dir: Override for the scenario files directory.
        progress: Optional ``TrainingProgress`` tracker.  The env owns
            episode-return / episode-length bookkeeping and emits events
            directly, so agents don't interact with progress at all.
    """

    def __init__(
        self,
        sequence: Any,
        scene_ids: list[str],
        run_ids: list[str],
        render_mode: str | None = None,
        render_speed: float = 1.0,
        stimuli_path: Path = STIMULI_PATH,
        scenarios_dir: Path = SCENARIOS_DIR,
        progress: Any = None,
    ) -> None:
        self._sequence_iter: Iterator = iter(sequence)
        self._clip_total: int | None = (
            len(sequence) if hasattr(sequence, "__len__") else None
        )
        self._scene_ids = scene_ids
        self._run_ids = run_ids
        self._render_mode = render_mode
        self._render_speed = render_speed
        self._stimuli_path = stimuli_path
        self._scenarios_dir = scenarios_dir
        self._progress = progress

        # Loaded metadata for exit points.
        self._scene_metadata = load_metadata(scenarios_dir)

        # Active environment and tracking state.
        self._env: TaskIdWrapper | None = None
        self._current_scene_id: str | None = None
        self._current_run_id: str | None = None
        self._next_spec: Any = None  # prefetched from sequence
        self._current_spec: Any = None  # spec of the episode currently being played
        self._episode_count: int = 0
        self._done: bool = False
        self._current_session: str | None = None

        # Episode bookkeeping owned by the env (used to emit progress events
        # without involving agent code).
        self._episode_return: float = 0.0
        self._episode_len: int = 0
        self._global_step: int = 0
        self._pending_end: dict | None = None

        # Prefetch the first spec so we can set up observation/action spaces.
        self._prefetch_next()
        if self._next_spec is not None:
            self._build_env(
                self._next_spec.scene_id,
                self._next_spec.run_id,
            )

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
            - ``task_switch`` (bool): True if the BIDS run changed.
            - ``scene_switch`` (bool): True if the scene changed.
            - ``run_id`` / ``run_index`` (str/int): Current task identity.
            - ``scene_id`` (str): Current Mario scene.
            - ``episode_index`` (int): Global episode counter.
            - All fields from ``SceneEnv._add_episode_info()``.

        Raises:
            StopIteration: When the sequence is exhausted.
        """
        # Flush any pending episode-end event first so that the agent's
        # ``logger.store({...})`` (called between step() and reset()) has a
        # chance to push scalars into ``progress.extra_metrics`` before the
        # display renders the completed episode.
        self._flush_pending_end()

        if self._done or self._next_spec is None:
            raise StopIteration("Curriculum sequence is exhausted.")

        spec = self._next_spec
        self._current_spec = spec
        task_switch = spec.run_id != self._current_run_id
        scene_switch = spec.scene_id != self._current_scene_id

        # Rebuild the underlying SceneEnv whenever the scene changes (the
        # retro scenario file is scene-specific). The task one-hot on the
        # wrapper is updated even when the scene stays the same but the run
        # changes (e.g. a scene reappearing in a new task).
        if scene_switch:
            if self._env is not None:
                self._env.close()
            self._build_env(spec.scene_id, spec.run_id)
        elif task_switch:
            # Same scene, new run: just retag the one-hot.
            self._env.set_run_id(spec.run_id)  # type: ignore[union-attr]

        if task_switch:
            logger.info("Task switch → %s (scene=%s)", spec.run_id, spec.scene_id)
        elif scene_switch:
            logger.debug(
                "Scene switch within %s → %s", spec.run_id, spec.scene_id
            )

        self._current_run_id = spec.run_id
        self._current_scene_id = spec.scene_id

        obs, info = self._env.reset(episode_spec=spec, seed=seed)
        info["task_switch"] = task_switch
        info["scene_switch"] = scene_switch
        info["run_id"] = spec.run_id
        info["run_index"] = spec.run_index
        info["session"] = spec.session
        info["session_switch"] = (
            self._current_session is not None
            and spec.session != self._current_session
        )
        self._current_session = spec.session
        info["episode_index"] = self._episode_count

        # Expose spec-derived clip metadata so the progress tracker (and
        # anything else downstream) can render curriculum position without
        # reaching into _current_spec.
        info["clip_code"] = spec.clip_code
        info["clip_index"] = self._episode_count  # 0-based position in sequence
        info["clip_total"] = self._clip_total
        info["subject"] = spec.subject
        info["level"] = spec.level
        info["phase"] = spec.phase
        info["human_outcome"] = spec.outcome

        self._episode_count += 1
        self._prefetch_next()

        # Fresh per-episode counters (the env owns these so agents don't
        # have to track them for progress purposes).
        self._episode_return = 0.0
        self._episode_len = 0

        # Notify the progress tracker about the new episode.
        if self._progress is not None:
            self._progress.on_reset(info)

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

        # Episode bookkeeping for progress events.
        self._episode_return += float(reward)
        self._episode_len += 1
        self._global_step += 1

        if terminated or truncated:
            stats = self.episode_stats
            self._pending_end = {
                "episode_return": self._episode_return,
                "episode_len": self._episode_len,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "cleared": bool(getattr(stats, "cleared", False)),
                "outcome": getattr(stats, "outcome", None),
                "global_step": self._global_step,
            }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current frame."""
        if self._env is None:
            return None
        return self._env.render()

    def close(self) -> None:
        """Close the active environment."""
        # Flush the final episode's progress event so the display sees the
        # last clip before the emulator shuts down.
        self._flush_pending_end()
        if self._env is not None:
            self._env.close()

    def _flush_pending_end(self) -> None:
        """Fire a deferred ``on_episode_end`` event, if one is pending."""
        if self._pending_end is None:
            return
        if self._progress is not None:
            self._progress.on_episode_end(**self._pending_end)
        self._pending_end = None

    def release_emulator(self) -> None:
        """Temporarily close the inner scene env.

        stable-retro allows only one emulator per process, so this must be
        called before another component creates a separate retro env (e.g.
        agent burn-in on a different scene). Call ``reacquire_emulator()``
        afterward to rebuild the inner env at the current/pending scene.
        """
        if self._env is not None:
            self._env.close()
            self._env = None

    def reacquire_emulator(self) -> None:
        """Rebuild the inner scene env after ``release_emulator()``.

        Rebuilds at ``_current_scene_id`` if set, otherwise at the pending
        next scene from the sequence. No-op if the env is already built
        or no scene is available.
        """
        if self._env is not None:
            return
        scene_id = self._current_scene_id
        run_id = self._current_run_id
        if scene_id is None and self._next_spec is not None:
            scene_id = self._next_spec.scene_id
            run_id = self._next_spec.run_id
        if scene_id is None or run_id is None:
            return
        self._build_env(scene_id, run_id)

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
            run_id=self._current_run_id,
            run_ids=self._run_ids,
            spec=self._current_spec,
            stimuli_path=self._stimuli_path,
            scenarios_dir=self._scenarios_dir,
            render_speed=self._render_speed,
        )
        self._build_env(self._current_scene_id, self._current_run_id)

    def _prefetch_next(self) -> None:
        """Advance the sequence iterator by one step."""
        try:
            self._next_spec = next(self._sequence_iter)
        except StopIteration:
            self._next_spec = None
            self._done = True

    def _build_env(self, scene_id: str, run_id: str) -> None:
        """Create a fresh wrapped env for ``scene_id`` tagged with ``run_id``.

        Args:
            scene_id: Scene identifier to build the env for (drives emulator).
            run_id: Current task (run) identifier for the one-hot.
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
            run_id=run_id,
            run_ids=self._run_ids,
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
        run_ids: list[str],
        max_steps: int,
        render_mode: str | None = None,
        render_speed: float = 1.0,
        stimuli_path: Path = STIMULI_PATH,
        scenarios_dir: Path = SCENARIOS_DIR,
        progress: Any = None,
    ) -> None:
        super().__init__(
            sequence=sequence,
            scene_ids=scene_ids,
            run_ids=run_ids,
            render_mode=render_mode,
            render_speed=render_speed,
            stimuli_path=stimuli_path,
            scenarios_dir=scenarios_dir,
            progress=progress,
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
