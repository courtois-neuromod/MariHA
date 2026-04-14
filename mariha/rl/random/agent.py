"""Random-action agent — minimal BenchmarkAgent implementation.

``RandomAgent`` selects actions uniformly at random and never learns.  It
serves two purposes:

1. **Documentation template**: the simplest possible complete implementation
   of ``BenchmarkAgent``.  Copy this file, replace the random policy with
   your agent, and you have a working MariHA agent.

2. **Performance lower bound**: running the benchmark with
   ``--agent random`` produces a valid ``eval_results.json`` with AP,
   BWT, and behavioral metrics that anchor comparisons.

See ``docs/adding_an_agent.md`` for a step-by-step guide.
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
        run_ids: list[str],
        seed: int = 0,
        experiment_dir: Path = Path("experiments"),
        timestamp: str = "",
    ) -> None:
        self.env = env
        self.logger = logger
        self.run_ids = run_ids
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

        current_run_id: str = info.get("run_id", "")
        current_task_idx: int = (
            self.run_ids.index(current_run_id)
            if current_run_id in self.run_ids
            else 0
        )
        one_hot_vec = info["task_one_hot"]
        self.on_task_start(current_task_idx, current_run_id)

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
                self.logger.store(
                    {
                        "train/return": episode_return,
                        "train/ep_length": episode_len,
                        "train/episodes": episodes,
                    }
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
                    new_run_id = info.get("run_id", "")
                    new_task_idx = (
                        self.run_ids.index(new_run_id)
                        if new_run_id in self.run_ids
                        else current_task_idx
                    )
                    # Save a checkpoint for the completed task
                    task_dir = self._checkpoint_dir(current_task_idx)
                    self.save_checkpoint(task_dir)

                    current_task_idx = new_task_idx
                    current_run_id = new_run_id
                    self.on_task_start(current_task_idx, current_run_id)

        self.on_task_end(current_task_idx)
        self.save_checkpoint(self._checkpoint_dir(current_task_idx))
        self.logger.log(
            f"RandomAgent complete — {episodes} episodes.", color="green"
        )

    def save_checkpoint(self, directory: Path) -> None:
        """Save agent metadata to ``directory/agent.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"agent": "random", "n_actions": int(self.n_actions)}
        (directory / "agent.json").write_text(json.dumps(meta, indent=2))

    def load_checkpoint(self, directory: Path) -> None:
        """No-op: random agent has no learned weights to restore."""

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser) -> None:
        """RandomAgent has no agent-specific hyperparameters."""

    @classmethod
    def from_args(cls, args, env, logger, run_ids) -> "RandomAgent":
        from mariha.utils.running import get_readable_timestamp
        # Prefer the seeded timestamp the benchmark context already composed
        # so the checkpoint dir leaf shares its prefix with the run dir leaf.
        timestamp = getattr(args, "run_timestamp", None) or (
            f"{get_readable_timestamp()}_seed{getattr(args, 'seed', 0)}"
        )
        return cls(
            env=env,
            logger=logger,
            run_ids=run_ids,
            seed=getattr(args, "seed", 0),
            experiment_dir=Path(getattr(args, "experiment_dir", "experiments")),
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _checkpoint_dir(self, task_idx: int) -> Path:
        agent_name = "random"
        ts = self.timestamp or ""
        return (
            self.experiment_dir
            / "checkpoints"
            / agent_name
            / f"{ts}_task{task_idx}"
        )
