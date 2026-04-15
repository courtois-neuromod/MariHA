"""BenchmarkAgent: abstract base class for all MariHA agent implementations.

Any agent that wants to run on the MariHA benchmark must subclass
``BenchmarkAgent`` and implement its abstract methods.  The benchmark
infrastructure (curriculum, environment, evaluation, metrics) interacts with
agents exclusively through this interface.

Minimal contract
----------------
- ``get_action`` — select an action given an observation and task identity.
- ``run`` — execute the full training loop until the curriculum is exhausted.
- ``save_checkpoint`` / ``load_checkpoint`` — persist and restore model state.

Optional hooks
--------------
- ``on_task_start`` / ``on_task_end`` — called at task boundaries (no-ops by
  default; useful for EWC Fisher computation, PackNet pruning, etc.).

Config interface
----------------
- ``add_args`` — add agent-specific CLI flags to an argparse parser.
- ``from_args`` — construct an agent from parsed args + benchmark context.

Future extension
----------------
- ``get_named_parameter_groups`` — expose named parameter groups to enable
  agent-agnostic CL regularizers (not required now).
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class BenchmarkAgent(ABC):
    """Abstract base class every MariHA agent must implement."""

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action(
        self,
        obs: np.ndarray,
        task_one_hot: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Return an action integer given an observation and task identity.

        Args:
            obs: Current observation from the environment.
            task_one_hot: One-hot vector identifying the current scene/task.
                Agents that don't use task identity may ignore this.
            deterministic: If ``True``, return the greedy action (used during
                evaluation).  If ``False``, sample stochastically (training).

        Returns:
            An integer action index compatible with the environment's
            ``Discrete`` action space.
        """

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        """Execute the full training loop until the curriculum is exhausted.

        The environment and curriculum are passed at construction time (via
        ``from_args`` or directly).  This method should:

        - Step through the environment episode by episode.
        - Call ``on_task_start`` / ``on_task_end`` at task boundaries.
        - Call ``save_checkpoint`` periodically (at minimum once per task).
        - Log progress via the logger passed at construction time.
        """

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @abstractmethod
    def save_checkpoint(self, directory: Path) -> None:
        """Persist all state needed for inference to ``directory``.

        The benchmark dictates only the *directory* (via the ``task{k}/``
        naming convention); the internal file layout is entirely up to the
        agent.  At minimum, save whatever ``load_checkpoint`` needs to
        reconstruct the policy for evaluation.

        Args:
            directory: Target directory (will be created if absent).
        """

    @abstractmethod
    def load_checkpoint(self, directory: Path) -> None:
        """Restore state from a checkpoint written by ``save_checkpoint``.

        Args:
            directory: Checkpoint directory previously written by
                ``save_checkpoint``.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional — default no-ops)
    # ------------------------------------------------------------------

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        """Optional pre-training phase on a single scene.

        Repeatedly plays ``burn_in_spec`` until ``num_steps`` transitions
        are collected.  Default implementation is a no-op.

        Args:
            burn_in_spec: ``EpisodeSpec`` for the burn-in scene.
            num_steps: Total environment steps to collect.
        """

    def on_task_start(self, task_idx: int, run_id: str) -> None:
        """Called at the beginning of each new task.

        Override to perform task-boundary setup: reset optimizer, snapshot
        parameters (EWC), allocate new heads (multi-head), etc.

        Args:
            task_idx: Index of the new task in ``run_ids``.
            run_id: BIDS run identifier string (e.g. ``'ses-001_run-02'``).
        """

    def on_task_end(self, task_idx: int) -> None:
        """Called at the end of each task (before moving to the next).

        Override to perform end-of-task operations: compute Fisher information
        (EWC), prune network (PackNet), consolidate episodic memory, etc.

        Args:
            task_idx: Index of the task that just finished.
        """

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add agent-specific CLI flags to ``parser``.

        Called by ``run_benchmark.py`` after the benchmark-level flags have
        been added, so agent flags are parsed in the same ``parse_args``
        call.  Default implementation is a no-op.

        Args:
            parser: The argparse parser to augment.
        """

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        env,
        logger,
        run_ids: list,
    ) -> "BenchmarkAgent":
        """Construct an agent from parsed CLI args and benchmark context.

        Args:
            args: Parsed argparse namespace (benchmark + agent flags).
            env: A ``ContinualLearningEnv`` instance (or compatible env).
            logger: An ``EpochLogger`` instance.
            run_ids: Canonical ordered list of all BIDS run IDs.

        Returns:
            A fully constructed agent ready to call ``run()`` on.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_args()."
        )

    # ------------------------------------------------------------------
    # Future extension: agent-agnostic CL regularizers
    # ------------------------------------------------------------------

    def get_named_parameter_groups(self) -> dict[str, list]:
        """Return named groups of trainable parameters.

        Implement this to enable agent-agnostic CL regularizers (e.g. an
        ``EWCWrapper`` that works across SAC, DDQN, PPO, …).  Not required
        for the benchmark to function; only needed if you want to compose your
        agent with a generic CL method.

        Returns:
            Dict mapping group name (e.g. ``"trunk"``, ``"policy_head"``) to
            a list of trainable parameter tensors/variables.

        Raises:
            NotImplementedError: Default — agent-agnostic CL not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose named parameter groups. "
            "Implement get_named_parameter_groups() to enable agent-agnostic CL."
        )
