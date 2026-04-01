"""Scene-level environment: episode initialization and termination.

``SceneEnv`` wraps ``MarioEnv`` and is responsible for the episode lifecycle:

- Loading the human clip's ``.state`` file at the start of each episode.
- Enforcing the human-aligned frame budget (``max_steps``).
- Detecting success (X-position >= exit point) and death (lives lost).
- Tracking per-episode behavioral statistics for evaluation.

Episode termination conditions (whichever occurs first):
    - **Success** (``terminated=True``):  ``x_pos >= exit_point``
    - **Death**   (``terminated=True``):  lives decreased from the initial value
    - **Timeout** (``truncated=True``):   ``step_count >= max_steps``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mariha.env.base import MarioEnv, STIMULI_PATH, SCENARIOS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Per-episode behavioral statistics, mirroring the human events TSV schema.

    Attributes:
        scene_id: Scene identifier (e.g. ``'w1l1s0'``).
        cleared: Whether the agent reached the exit point.
        steps: Total steps taken in the episode.
        x_traveled: Net X-distance traveled (can be negative if Mario moves left).
        score_gained: Score delta over the episode.
        coins_gained: Coins collected.
        lives_lost: 1 if the episode ended in death, 0 otherwise.
        outcome: Human-readable outcome string (``'completed'`` or ``'failed'``).
    """

    scene_id: str
    cleared: bool = False
    steps: int = 0
    x_traveled: int = 0
    score_gained: int = 0
    coins_gained: int = 0
    lives_lost: int = 0
    outcome: str = "failed"

    def to_dict(self) -> dict[str, Any]:
        """Return stats as a plain dictionary."""
        return {
            "scene_id": self.scene_id,
            "cleared": self.cleared,
            "steps": self.steps,
            "x_traveled": self.x_traveled,
            "score_gained": self.score_gained,
            "coins_gained": self.coins_gained,
            "lives_lost": self.lives_lost,
            "outcome": self.outcome,
        }


# ---------------------------------------------------------------------------
# SceneEnv
# ---------------------------------------------------------------------------


class SceneEnv:
    """Episode-level environment driven by a human clip EpisodeSpec.

    Wraps ``MarioEnv`` and enforces the human-aligned training budget and
    all termination conditions.  Call ``reset(episode_spec)`` at the start
    of each episode to load the corresponding human ``.state`` file.

    Args:
        scene_id: Scene identifier (e.g. ``'w1l1s0'``).
        exit_point: Absolute X-position at which the scene is considered
            cleared (from ``scenes_mastersheet.csv``).
        render_mode: Passed through to ``MarioEnv``.
        stimuli_path: Override for the stimuli directory path.
        scenarios_dir: Override for the scenario files directory.

    Attributes:
        observation_space: Observation space of the underlying env.
        action_space: Action space of the underlying env.
        episode_stats: Statistics from the most recently completed episode.
    """

    def __init__(
        self,
        scene_id: str,
        exit_point: int,
        render_mode: str | None = None,
        stimuli_path: Path = STIMULI_PATH,
        scenarios_dir: Path = SCENARIOS_DIR,
    ) -> None:
        self.scene_id = scene_id
        self.exit_point = exit_point

        self._base = MarioEnv(
            scene_id=scene_id,
            render_mode=render_mode,
            stimuli_path=stimuli_path,
            scenarios_dir=scenarios_dir,
        )
        self.observation_space = self._base.observation_space
        self.action_space = self._base.action_space

        # Episode state
        self._max_steps: int = 0
        self._step_count: int = 0
        self._initial_lives: int = 0
        self._initial_x: int = 0
        self._initial_score: int = 0
        self._initial_coins: int = 0

        self.episode_stats: EpisodeStats = EpisodeStats(scene_id=scene_id)

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        episode_spec: "EpisodeSpec",  # noqa: F821 — forward ref, imported at runtime
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode.

        Loads the human clip ``.state`` file from ``episode_spec`` and
        records the initial lives, score, and position for tracking.

        Args:
            episode_spec: The ``EpisodeSpec`` describing this episode
                (state file, max steps, scene metadata).
            seed: Optional RNG seed.

        Returns:
            ``(observation, info)`` tuple.
        """
        self._max_steps = episode_spec.max_steps
        self._step_count = 0
        self.episode_stats = EpisodeStats(scene_id=self.scene_id)

        obs, info = self._base.reset(
            state_file=Path(episode_spec.state_file),
            seed=seed,
        )

        # Snapshot initial values for delta computations.
        self._initial_lives = int(info.get("lives", 2))
        self._initial_x = int(info.get("x_pos", 0))
        self._initial_score = int(info.get("score", 0))
        self._initial_coins = int(info.get("coins", 0))

        info = self._add_episode_info(info)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment.

        Checks all three termination conditions after each step.

        Args:
            action: Action to execute (forwarded to the base env).

        Returns:
            ``(obs, reward, terminated, truncated, info)`` tuple where:
            - ``terminated`` is ``True`` on success or death.
            - ``truncated`` is ``True`` on timeout.
        """
        obs, reward, _base_term, _base_trunc, info = self._base.step(action)
        self._step_count += 1

        x_pos = int(info["x_pos"])
        lives = int(info.get("lives", self._initial_lives))

        # --- Termination checks ---
        success = x_pos >= self.exit_point
        death = lives < self._initial_lives
        timeout = self._step_count >= self._max_steps

        terminated = success or death
        truncated = (not terminated) and timeout

        # --- Update running stats on terminal step ---
        if terminated or truncated:
            self._update_final_stats(info, success)

        info = self._add_episode_info(info)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current frame."""
        return self._base.render()

    def close(self) -> None:
        """Close the environment."""
        self._base.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_final_stats(self, info: dict, success: bool) -> None:
        """Populate ``episode_stats`` at the end of an episode.

        Args:
            info: Final info dict from the last ``step()``.
            success: Whether the episode ended in scene completion.
        """
        x_pos = int(info.get("x_pos", self._initial_x))
        score = int(info.get("score", self._initial_score))
        coins = int(info.get("coins", self._initial_coins))
        lives = int(info.get("lives", self._initial_lives))

        self.episode_stats.cleared = success
        self.episode_stats.steps = self._step_count
        self.episode_stats.x_traveled = x_pos - self._initial_x
        self.episode_stats.score_gained = score - self._initial_score
        self.episode_stats.coins_gained = coins - self._initial_coins
        self.episode_stats.lives_lost = max(0, self._initial_lives - lives)
        self.episode_stats.outcome = "completed" if success else "failed"

    def _add_episode_info(self, info: dict) -> dict:
        """Append episode-level fields to the info dict.

        Args:
            info: Info dict to augment.

        Returns:
            Augmented info dict with ``step_count``, ``max_steps``,
            ``exit_point``, and ``scene_id`` fields.
        """
        enriched = dict(info)
        enriched["step_count"] = self._step_count
        enriched["max_steps"] = self._max_steps
        enriched["exit_point"] = self.exit_point
        enriched["scene_id"] = self.scene_id
        return enriched

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        """The raw stable-retro environment."""
        return self._base.unwrapped
