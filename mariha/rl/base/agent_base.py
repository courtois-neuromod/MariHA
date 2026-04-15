"""BaseAgent: shared mixin for MariHA RL algorithms.

:class:`BaseAgent` extends :class:`BenchmarkAgent` with a callback-style
interface and delegates the episode-driven training loop to
:class:`TrainingLoopRunner`.  Concrete agents (SAC, PPO, DQN, future
additions like MuZero or LLM-based policies) implement only the
algorithm-specific callbacks listed below; everything else — episode
bookkeeping, task switches, session flushes, render checkpoints,
logging, periodic saves — is provided here.

**Required callbacks** (abstract)
    - :meth:`get_action` — greedy evaluation action (BenchmarkAgent contract).
    - :meth:`select_action` — training action, possibly with exploration
      noise / epsilon-greedy / stochastic sampling.  Returns
      ``(action, extras)`` where ``extras`` is a dict of per-transition
      side data the agent forwards to its own :meth:`store_transition`
      (PPO uses this to carry ``value`` and ``log_prob``).
    - :meth:`store_transition` — write the transition to the agent's
      replay or rollout buffer.
    - :meth:`should_update` — given ``global_step``, return whether an
      update should fire.  Per-step agents return True on an
      ``update_every`` cadence; per-rollout agents return True only when
      their rollout buffer is full.
    - :meth:`update_step` — one round of gradient updates.
    - :meth:`save_weights` / :meth:`load_weights` — per-task checkpoint I/O.

**Optional hooks** (default implementations provided)
    - :meth:`on_task_change` — per-task resets (buffer, network,
      optimizer).  Called after ``on_task_start``; default is a no-op.
    - :meth:`handle_session_boundary` — per-scene buffer flush logic.
    - :meth:`on_episode_end` — logs return/length/episode count.
    - :meth:`log_after_epoch` — writes the standard tabular row plus any
      agent-specific keys returned by :meth:`get_log_tabular_keys`.
    - :meth:`on_burn_in_start` / :meth:`on_burn_in_end` — optional
      bookends around :func:`run_burn_in`.

**Class configuration**
    - :attr:`update_granularity` — informational tag (``"per_step"`` or
      ``"per_rollout"``).  The runner does not branch on this, but the
      attribute documents the agent's style for the developer.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from mariha.benchmark.agent import BenchmarkAgent
from mariha.rl.base.checkpoint import standard_checkpoint_dir
from mariha.utils.logging import EpochLogger
from mariha.utils.running import set_seed

if TYPE_CHECKING:
    from mariha.methods.base import CLMethod


class BaseAgent(BenchmarkAgent):
    """Shared base class for MariHA RL agents.

    See the module docstring for the full callback surface and lifecycle.
    """

    #: Informational tag describing the update pattern.  ``"per_step"``
    #: (SAC, DQN) means updates fire on a per-step schedule via
    #: ``should_update``.  ``"per_rollout"`` (PPO) means the agent
    #: collects a full rollout before ``should_update`` returns True.
    #: The runner does not branch on this attribute — it exists to
    #: document the agent's style for readers.
    update_granularity: str = "per_step"

    def __init__(
        self,
        env,
        logger: EpochLogger,
        run_ids: List[str],
        *,
        agent_name: str,
        seed: int = 0,
        log_every: int = 1000,
        save_freq_epochs: int = 25,
        render_every: int = 0,
        experiment_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Initialise shared agent state.

        Subclasses forward their algorithm-agnostic args via
        ``super().__init__(...)`` and then set up their own networks,
        buffers, optimizers, etc.  The global seed is applied exactly
        once here — subclasses must not call :func:`set_seed` again.

        Args:
            env: The :class:`ContinualLearningEnv` instance.  Both its
                action and observation spaces must already be populated.
            logger: An :class:`EpochLogger` — used for progress messages
                and for the standard epoch tabular row.
            run_ids: Ordered list of all BIDS run IDs in the curriculum
                (one per continual-learning task).  Defines the one-hot
                dimension and the mapping from ``run_id`` strings to
                integer task indices.
            agent_name: Short identifier used for checkpoint paths and
                log messages (e.g. ``"sac"``, ``"ppo"``, ``"dqn"``).
            seed: Global random seed applied to Python, NumPy, TF and
                the environment via :func:`set_seed`.
            log_every: Number of environment steps per logging epoch.
            save_freq_epochs: Checkpoint cadence in logging epochs.
            render_every: If > 0, play a live greedy episode every N
                training episodes.  0 = disabled.
            experiment_dir: Root directory for checkpoints and log
                output.  Defaults to ``./experiments``.
            timestamp: Run timestamp string — used to disambiguate
                checkpoint directories for concurrent runs.
        """
        set_seed(seed, env=env)

        self.env = env
        self.logger = logger
        self.run_ids = run_ids
        self.num_tasks = len(run_ids)
        self.agent_name = agent_name
        self.log_every = int(log_every)
        self.save_freq_epochs = int(save_freq_epochs)
        self.render_every = int(render_every)
        self.experiment_dir = Path(experiment_dir or "experiments")
        self.timestamp = timestamp or ""

        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n

        logger.log(f"Observation shape: {self.obs_shape}", color="blue")
        logger.log(f"Action dim:        {self.act_dim}", color="blue")
        logger.log(f"Num tasks:         {self.num_tasks}", color="blue")

        self.start_time: float = 0.0

        #: Composition slot for a continual learning method.  Set after
        #: agent construction (typically by the runner script) via
        #: ``agent.cl_method = SomeCLMethod(...)``.  When ``None`` the
        #: agent runs in vanilla mode with no CL augmentation.  See
        #: :class:`mariha.methods.base.CLMethod`.
        self.cl_method: Optional["CLMethod"] = None

    # ==================================================================
    # Required callback surface — subclasses MUST implement
    # ==================================================================

    @abstractmethod
    def select_action(
        self,
        *,
        obs: np.ndarray,
        one_hot: np.ndarray,
        global_step: int,
        task_step: int,
        current_task_idx: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """Return ``(action, extras)`` for one training step.

        ``extras`` is an arbitrary dict the agent forwards to its own
        :meth:`store_transition`.  Off-policy agents typically return
        ``{}``; PPO returns ``{"log_prob": ..., "value": ...}``.

        Note that both ``global_step`` (over the whole curriculum) and
        ``task_step`` (reset on every task switch) are provided.  The
        agent can use whichever it needs — SAC uses ``task_step`` for
        its ``start_steps`` warm-up, DQN uses ``global_step`` for
        epsilon decay, PPO ignores both.
        """

    @abstractmethod
    def store_transition(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        one_hot: np.ndarray,
        scene_id: str,
        run_id: str,
        info: Dict[str, Any],
        extras: Dict[str, Any],
    ) -> None:
        """Store one transition in the agent's replay or rollout buffer.

        All of the loop's per-step state is made available; the agent
        uses whatever it needs and ignores the rest.  Off-policy agents
        typically use ``obs, action, reward, next_obs, terminated,
        one_hot`` (and ``scene_id`` in per-scene buffer mode).  On-policy
        agents like PPO read ``extras["value"]`` and ``extras["log_prob"]``
        for the rollout buffer and additionally cache ``next_obs`` /
        ``one_hot`` for the post-rollout bootstrap.
        """

    @abstractmethod
    def should_update(self, global_step: int) -> bool:
        """Return True if :meth:`update_step` should fire at this step."""

    @abstractmethod
    def update_step(
        self, *, global_step: int, current_task_idx: int
    ) -> None:
        """Perform one round of gradient updates.

        Off-policy agents (SAC, DQN) sample a batch from the replay
        buffer and apply one or more gradient steps.  On-policy agents
        (PPO) bootstrap the rollout's final value, compute returns and
        advantages via GAE, run ``n_epochs`` of minibatch updates, and
        reset their rollout buffer.
        """

    @abstractmethod
    def save_weights(self, directory: Path) -> None:
        """Save learned parameters to ``directory``.

        ``directory`` already exists when this is called.  The agent is
        free to use any file layout it wants inside.
        """

    @abstractmethod
    def load_weights(self, directory: Path) -> None:
        """Load learned parameters previously saved by :meth:`save_weights`."""

    # ==================================================================
    # Reference-gradient hook (used by AGEM, distillation methods)
    # ==================================================================

    def _compute_reference_loss(
        self, batch: Dict[str, "tf.Tensor"]
    ) -> Dict[str, "tf.Tensor"]:
        """Return per-group scalar losses on ``batch`` (no CL penalty).

        Subclasses override to return one differentiable scalar per
        parameter group in :meth:`get_named_parameter_groups`, computed
        from the agent's standard training loss but **without** the
        attached CL method's penalty.  The returned dict keys must
        match the group names so :meth:`compute_reference_gradients`
        can pair losses with their respective parameter lists.

        For SAC: ``{"actor": actor_loss, "critic1": q1_loss,
        "critic2": q2_loss}``.  For PPO: ``{"policy": total_loss}``.
        For DQN: ``{"q": td_loss}``.

        Default raises ``NotImplementedError`` — only required for
        agents that want to support gradient-projection methods like
        AGEM.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement "
            f"_compute_reference_loss; gradient-projection CL methods "
            f"like AGEM are not supported on this agent."
        )

    def compute_reference_gradients(
        self, batch: Dict[str, "tf.Tensor"]
    ) -> Dict[str, List["tf.Tensor"]]:
        """Compute reference gradients on ``batch`` for all parameter groups.

        Returns a dict mapping group name → list of gradients (one per
        parameter, in the same order as
        :meth:`get_named_parameter_groups`).  Used by AGEM to compute
        the constraint gradient on its episodic memory.

        This is a template method: subclasses normally override
        :meth:`_compute_reference_loss` instead.  The body sets up a
        persistent ``GradientTape``, calls
        :meth:`_compute_reference_loss` to get per-group scalar
        losses, and differentiates each w.r.t. its corresponding
        parameter group.
        """
        import tensorflow as tf

        groups = self.get_named_parameter_groups()
        with tf.GradientTape(persistent=True) as tape:
            losses_by_group = self._compute_reference_loss(batch)
        grads_by_group: Dict[str, List["tf.Tensor"]] = {}
        for gn, loss in losses_by_group.items():
            if gn not in groups:
                continue
            grads = tape.gradient(loss, groups[gn])
            grads_by_group[gn] = list(grads)
        del tape
        return grads_by_group

    # ==================================================================
    # Optional callbacks — defaults provided
    # ==================================================================

    def on_task_start(self, task_idx: int, run_id: str) -> None:
        """Forward the task-start signal to the attached CL method.

        Subclasses that override this for their own logging or setup
        should call ``super().on_task_start(task_idx, run_id)`` so the
        ``cl_method`` hook still fires.  The default implementation
        does nothing else.
        """
        if self.cl_method is not None:
            self.cl_method.on_task_start(self, task_idx)

    def on_task_end(self, task_idx: int) -> None:
        """Forward the task-end signal to the attached CL method.

        Fires *before* :meth:`on_task_change` resets the agent's
        per-task state, so a regularizer hooked here can still read
        the just-finished task's data from the replay buffer.
        Subclasses that override this should call
        ``super().on_task_end(task_idx)`` so the ``cl_method`` hook
        still fires.
        """
        if self.cl_method is not None:
            self.cl_method.on_task_end(self, task_idx)

    def on_task_change(self, new_task_idx: int, new_run_id: str) -> None:
        """Reset agent state at a task boundary.

        Called by the runner after ``on_task_start(new_task_idx,
        new_run_id)`` — i.e. the agent already knows the new task has
        begun.  The default is a no-op.  Agents typically override this
        to:

        - Re-initialise the replay buffer (``reset_buffer_on_task_change``)
        - Re-initialise actor/critic weights (``reset_*_on_task_change``)
        - Reset optimizer slot variables
          (``reset_optimizer_on_task_change``)
        - Recompile a per-task ``tf.function``
        """

    def handle_session_boundary(self, current_task_idx: int) -> None:
        """Called when the curriculum crosses a session boundary.

        Used by agents with a per-scene buffer pool to flush pending
        transitions — typically by running gradient updates on the
        accumulated data and then emptying the scene buffers.  Default
        is a no-op.
        """

    def on_episode_end(
        self,
        *,
        episode_return: float,
        episode_len: int,
        total_episodes: int,
    ) -> None:
        """Log standard per-episode metrics.

        Default implementation stores ``train/return``,
        ``train/ep_length``, and ``train/episodes`` via
        :meth:`EpochLogger.store`.  Override to add agent-specific
        metrics (e.g. DQN appends ``train/epsilon`` and
        ``buffer_fill_pct``).
        """
        self.logger.store(
            {
                "train/return": float(episode_return),
                "train/ep_length": int(episode_len),
                "train/episodes": int(total_episodes),
            }
        )

    def get_log_tabular_keys(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return agent-specific tabular log keys as ``(key, kwargs)`` pairs.

        Called by the default :meth:`log_after_epoch` to append
        algorithm-specific metrics (losses, Q-values, entropy, etc.) to
        the standard row.  Example::

            return [
                ("train/loss_q", {"average_only": True}),
                ("train/q_mean", {"average_only": True}),
                ("train/epsilon", {"average_only": True}),
                ("buffer_fill_pct", {"average_only": True}),
            ]

        The keys must already have been populated in ``logger.store``
        during the epoch (typically inside :meth:`update_step` or
        :meth:`on_episode_end`).
        """
        return []

    def log_after_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        action_counts: Dict[int, int],
    ) -> None:
        """Write a tabular log row for the completed epoch.

        Default implementation logs a common core set of metrics
        (``epoch``, ``total_env_steps``, ``train/return`` with min/max,
        ``train/ep_length``, ``train/episodes``, ``walltime``) plus the
        per-action counts and any extra entries returned by
        :meth:`get_log_tabular_keys`.  Override for custom row layouts.
        """
        self.logger.log_tabular("epoch", epoch)
        self.logger.log_tabular("total_env_steps", global_step + 1)
        self.logger.log_tabular("train/return", with_min_and_max=True)
        self.logger.log_tabular("train/ep_length", average_only=True)
        self.logger.log_tabular("train/episodes", average_only=True)

        for key, kwargs in self.get_log_tabular_keys():
            self.logger.log_tabular(key, **kwargs)

        self._log_action_counts(action_counts)
        self.logger.log_tabular("walltime", time.time() - self.start_time)
        self.logger.dump_tabular()

    def _log_action_counts(self, action_counts: Dict[int, int]) -> None:
        """Log per-action frequency counts.

        Uses :data:`ACTION_NAMES` when available so the tabular row is
        readable; falls back to integer indices if the MariHA action
        wrapper isn't importable (e.g. in synthetic unit tests).
        """
        try:
            from mariha.env.wrappers.action import ACTION_NAMES
        except ImportError:
            ACTION_NAMES = []

        for i, count in action_counts.items():
            name = (
                ACTION_NAMES[i]
                if ACTION_NAMES and i < len(ACTION_NAMES)
                else str(i)
            )
            self.logger.log_tabular(f"train/actions/{name}", count)

    def on_burn_in_start(self, task_idx: int) -> None:
        """Optional hook fired at the start of :func:`run_burn_in`.

        Override if the agent needs per-task setup for burn-in (e.g.
        SAC recompiles its ``learn_on_batch`` ``tf.function`` for the
        burn-in task).
        """

    def on_burn_in_end(self) -> None:
        """Optional hook fired after burn-in completes.

        Override to flush replay/rollout buffers and reset any
        burn-in-specific schedule state (typically, promoting
        ``post_burn_in_update_after`` into ``update_after``).
        """

    # ==================================================================
    # BenchmarkAgent contract — provided in terms of the callback surface
    # ==================================================================

    def run(self) -> None:
        """Execute the full curriculum via :class:`TrainingLoopRunner`."""
        # Imported lazily so that importing :mod:`agent_base` alone
        # doesn't pull in the runner (avoids any circular-import snags
        # while keeping ``BaseAgent`` available as a top-level symbol).
        from mariha.rl.base.training_loop import TrainingLoopRunner

        runner = TrainingLoopRunner(self)
        runner.run()

    def save_checkpoint(self, directory: Path) -> None:
        """BenchmarkAgent interface: delegates to :meth:`save_weights`.

        Ensures the target directory exists before writing.  Subclasses
        typically do not override this — they customise I/O layout by
        overriding :meth:`save_weights` instead.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_weights(directory)

    def load_checkpoint(self, directory: Path) -> None:
        """BenchmarkAgent interface: delegates to :meth:`load_weights`."""
        self.load_weights(Path(directory))

    def _checkpoint_dir(self, task_idx: int) -> Path:
        """Return the canonical per-task checkpoint directory.

        Identical to the directory the runner writes into during
        periodic saves; exposed here so agents (or tests) can
        reconstruct it without reaching into the runner's internals.
        """
        return standard_checkpoint_dir(
            self.experiment_dir,
            self.agent_name,
            self.timestamp,
            task_idx,
        )
