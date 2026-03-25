"""Random-action agent — minimal BenchmarkAgent implementation.

``RandomAgent`` selects actions uniformly at random and never learns.  It
serves two purposes:

1. **Documentation template**: the simplest possible complete implementation
   of ``BenchmarkAgent``.  Copy this file, replace the random policy with
   your algorithm, and you have a working MariHA agent.

2. **Performance lower bound**: running the benchmark with
   ``--algorithm random`` produces a valid ``eval_results.json`` with AP,
   BWT, and behavioral metrics that anchor comparisons.

See ``docs/adding_an_algorithm.md`` for a step-by-step guide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mariha.benchmark.agent import BenchmarkAgent
from mariha.utils.logging import EpochLogger


class RandomAgent(BenchmarkAgent):
    """Selects actions uniformly at random.  No learning occurs."""

    def __init__(
        self,
        env,
        logger: EpochLogger,
        scene_ids: list[str],
        seed: int = 0,
        experiment_dir: Path = Path("experiments"),
        timestamp: str = "",
    ) -> None:
        self.env = env
        self.logger = logger
        self.scene_ids = scene_ids
        self.n_actions = env.action_space.n
        self.rng = np.random.default_rng(seed)
        self.experiment_dir = experiment_dir
        self.timestamp = timestamp

    # ------------------------------------------------------------------
    # BenchmarkAgent interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        obs: np.ndarray,
        task_one_hot: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Return a uniformly random action (obs and task identity ignored)."""
        return int(self.rng.integers(self.n_actions))

    def run(self) -> None:
        """Consume the full curriculum, selecting random actions throughout."""
        try:
            obs, info = self.env.reset()
        except StopIteration:
            self.logger.log("Curriculum is empty — nothing to train.", color="red")
            return

        current_scene_id: str = info.get("scene_id", "")
        current_task_idx: int = (
            self.scene_ids.index(current_scene_id)
            if current_scene_id in self.scene_ids
            else 0
        )
        one_hot_vec = info["task_one_hot"]
        self.on_task_start(current_task_idx, current_scene_id)

        episode_return = 0.0
        episode_len = 0
        episodes = 0

        self.logger.log("RandomAgent training started.", color="green")

        while True:
            action = self.get_action(obs, one_hot_vec)
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_return += reward
            episode_len += 1

            if terminated or truncated:
                episodes += 1
                self.logger.log(
                    f"Ep {episodes:5d} | return={episode_return:7.2f} | len={episode_len:4d}"
                )
                episode_return = 0.0
                episode_len = 0

                if self.env.is_done:
                    break

                try:
                    obs, info = self.env.reset()
                except StopIteration:
                    break

                one_hot_vec = info["task_one_hot"]

                if info.get("task_switch", False):
                    self.on_task_end(current_task_idx)
                    new_scene_id = info.get("scene_id", "")
                    new_task_idx = (
                        self.scene_ids.index(new_scene_id)
                        if new_scene_id in self.scene_ids
                        else current_task_idx
                    )
                    # Save a checkpoint for the completed task
                    task_dir = self._checkpoint_dir(current_task_idx)
                    self.save_checkpoint(task_dir)

                    current_task_idx = new_task_idx
                    current_scene_id = new_scene_id
                    self.on_task_start(current_task_idx, current_scene_id)

        self.on_task_end(current_task_idx)
        self.save_checkpoint(self._checkpoint_dir(current_task_idx))
        self.logger.log(
            f"RandomAgent complete — {episodes} episodes.", color="green"
        )

    def save_checkpoint(self, directory: Path) -> None:
        """Save agent metadata to ``directory/agent.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"algorithm": "random", "n_actions": int(self.n_actions)}
        (directory / "agent.json").write_text(json.dumps(meta, indent=2))

    def load_checkpoint(self, directory: Path) -> None:
        """No-op: random agent has no learned weights to restore."""

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser) -> None:
        """RandomAgent has no algorithm-specific hyperparameters."""

    @classmethod
    def from_args(cls, args, env, logger, scene_ids) -> "RandomAgent":
        from mariha.utils.running import get_readable_timestamp
        return cls(
            env=env,
            logger=logger,
            scene_ids=scene_ids,
            seed=getattr(args, "seed", 0),
            experiment_dir=Path(getattr(args, "experiment_dir", "experiments")),
            timestamp=get_readable_timestamp(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _checkpoint_dir(self, task_idx: int) -> Path:
        algorithm = "random"
        ts = self.timestamp or ""
        return (
            self.experiment_dir
            / "checkpoints"
            / algorithm
            / f"{ts}_task{task_idx}"
        )
