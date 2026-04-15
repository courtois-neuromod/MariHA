"""End-to-end integration test for DQN on a synthetic curriculum.

The test plugs the migrated :class:`DQN` into the shared
:class:`TrainingLoopRunner` via :meth:`BaseAgent.run`, drives it through
a tiny image-shaped curriculum, and asserts that the loop produces
checkpoints and the expected log keys.  It is the Phase 1 acceptance
gate for the BaseAgent migration: it does not measure scientific
quality (no return-curve comparison), only that the wiring is intact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from mariha.rl.dqn import DQN


# ----------------------------------------------------------------------
# Synthetic curriculum environment
# ----------------------------------------------------------------------


class SyntheticCurriculumEnv:
    """Minimal continual-learning env for tests.

    Replays a hard-coded list of episodes through the standard env API
    used by :class:`TrainingLoopRunner`.  Observations are zero-filled
    image tensors of the requested shape so the real DQN CNN backbone
    can be exercised.
    """

    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        obs_shape=(84, 84, 4),
        n_actions: int = 7,
        num_tasks: int = 2,
    ) -> None:
        self.episodes = episodes
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(n_actions)
        self.num_tasks = num_tasks
        self.is_done = False

        self._idx = -1
        self._step_in_ep = 0

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _one_hot(self, scene_id: str) -> np.ndarray:
        v = np.zeros(self.num_tasks, dtype=np.float32)
        try:
            idx = int(scene_id.replace("scene", ""))
            if 0 <= idx < self.num_tasks:
                v[idx] = 1.0
        except ValueError:
            pass
        return v

    def reset(self, episode_spec=None):
        self._idx += 1
        if self._idx >= len(self.episodes):
            raise StopIteration
        ep = self.episodes[self._idx]
        self._step_in_ep = 0
        info = {
            "task_one_hot": self._one_hot(ep["scene_id"]),
            "scene_id": ep["scene_id"],
            "run_id": ep.get("run_id", ep["scene_id"]),
            "session": ep.get("session", "sess0"),
            "task_switch": ep.get("task_switch", False) if self._idx > 0 else False,
            "session_switch": ep.get("session_switch", False)
            if self._idx > 0
            else False,
        }
        return self._zero_obs(), info

    def step(self, action: int):
        self._step_in_ep += 1
        ep = self.episodes[self._idx]
        terminated = self._step_in_ep >= ep["length"]
        if terminated and self._idx == len(self.episodes) - 1:
            self.is_done = True
        return self._zero_obs(), 0.0, terminated, False, {}

    def render_checkpoint(self, actor_fn) -> None:
        # No-op for tests; render_every is set to 0 below.
        pass


# ----------------------------------------------------------------------
# Test logger that captures the keys actually written to the tabular row
# ----------------------------------------------------------------------


class CapturingLogger:
    """Drop-in replacement for :class:`EpochLogger` that records every
    ``log_tabular`` call so the test can assert which keys were emitted."""

    def __init__(self) -> None:
        self.messages: List[str] = []
        self.stored: List[Dict[str, Any]] = []
        self.tabular_keys_per_epoch: List[List[str]] = []
        self._current_epoch_keys: List[str] = []
        self.dumps: int = 0

    def log(self, msg: str, color: str = None) -> None:
        self.messages.append(msg)

    def store(self, d: Dict[str, Any]) -> None:
        self.stored.append(dict(d))

    def log_tabular(self, key: str, value=None, **kwargs) -> None:
        self._current_epoch_keys.append(key)

    def dump_tabular(self) -> None:
        self.tabular_keys_per_epoch.append(list(self._current_epoch_keys))
        self._current_epoch_keys = []
        self.dumps += 1


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def _build_curriculum() -> List[Dict[str, Any]]:
    """5 episodes × 20 steps each = 100 env steps; one task switch."""
    return [
        {"scene_id": "scene0", "session": "sess0", "length": 20,
         "task_switch": False, "session_switch": False},
        {"scene_id": "scene0", "session": "sess0", "length": 20,
         "task_switch": False, "session_switch": False},
        {"scene_id": "scene1", "session": "sess1", "length": 20,
         "task_switch": True, "session_switch": True},
        {"scene_id": "scene1", "session": "sess1", "length": 20,
         "task_switch": False, "session_switch": False},
        {"scene_id": "scene1", "session": "sess1", "length": 20,
         "task_switch": False, "session_switch": False},
    ]


def test_dqn_run_synthetic_curriculum():
    """DQN.run() should drive a 5-episode curriculum to completion,
    write checkpoints under the standard layout, and emit the expected
    DQN-specific log keys."""

    env = SyntheticCurriculumEnv(_build_curriculum())
    logger = CapturingLogger()

    with tempfile.TemporaryDirectory() as tmp:
        dqn = DQN(
            env=env,
            logger=logger,
            run_ids=["scene0", "scene1"],
            seed=0,
            replay_size=1000,
            update_after=16,
            update_every=4,
            batch_size=8,
            n_updates=1,
            log_every=50,
            save_freq_epochs=1,
            render_every=0,
            experiment_dir=Path(tmp),
            timestamp="test",
        )
        dqn.run()

        # Curriculum drove to completion: 5 episodes × 20 steps.
        assert env.is_done
        # Two log epochs (100 steps / log_every=50)
        assert logger.dumps == 2

        # Standard keys + DQN-specific keys are present in the dumped row.
        keys_epoch_0 = set(logger.tabular_keys_per_epoch[0])
        for required in [
            "epoch",
            "total_env_steps",
            "train/return",
            "train/ep_length",
            "train/episodes",
            "train/epsilon",
            "buffer_fill_pct",
            "train/loss_q",
            "train/q_mean",
            "walltime",
        ]:
            assert required in keys_epoch_0, f"missing tabular key: {required}"

        # Two checkpoint dirs under the standard layout — one per task.
        ckpt_root = Path(tmp) / "checkpoints" / "dqn"
        assert ckpt_root.exists(), "checkpoint root not created"
        ckpt_dirs = sorted(p.name for p in ckpt_root.iterdir())
        # save_freq_epochs=1 → at least the periodic save dirs exist;
        # final save also fires after the loop.
        assert len(ckpt_dirs) >= 1
        # The standard naming includes the task index suffix.
        assert any("task0" in d for d in ckpt_dirs) or any(
            "task1" in d for d in ckpt_dirs
        )
