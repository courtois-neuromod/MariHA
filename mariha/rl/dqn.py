"""Deep Q-Network (with optional Double DQN) for the MariHA benchmark.

Off-policy, replay-buffer-based, per-step updates with epsilon-greedy
exploration.  Built on :class:`mariha.rl.base.BaseAgent`, so episode
bookkeeping, task-switch handling, session-boundary flushing, render
checkpoints, periodic logging, and checkpointing all come from the
shared :class:`TrainingLoopRunner`.  This file contains only the DQN
algorithm itself plus the algorithm-specific lifecycle hooks.

Usage::

    mariha-run-cl --agent dqn  --subject sub-01 --seed 0 --lr 1e-4
    mariha-run-cl --agent ddqn --subject sub-01 --seed 0 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

from mariha.replay.buffers import BufferType, ReplayBuffer, ReservoirReplayBuffer
from mariha.rl import models
from mariha.rl.base import BaseAgent, run_burn_in
from mariha.utils.logging import EpochLogger


class DQN(BaseAgent):
    """Deep Q-Network for continual learning on MariHA.

    Uses :class:`models.MlpCritic` (the same architecture SAC uses for
    its critics) for the online and target Q-networks.  When
    ``double_dqn=True`` (default), action selection at the bootstrap
    target uses the online network and Q-value evaluation uses the
    target network.
    """

    update_granularity: str = "per_step"

    def __init__(
        self,
        env,
        logger: EpochLogger,
        run_ids: List[str],
        *,
        seed: int = 0,
        # ---- DQN-specific ----
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50_000,
        double_dqn: bool = True,
        replay_size: int = 100_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-4,
        batch_size: int = 32,
        update_after: int = 5_000,
        update_every: int = 4,
        n_updates: int = 1,
        # ---- buffer ----
        buffer_mode: str = "single",
        buffer_type: BufferType = BufferType.FIFO,
        per_scene_capacity: int = 1000,
        flush_on: str = "session",
        reset_buffer_on_task_change: bool = True,
        reset_optimizer_on_task_change: bool = False,
        reset_network_on_task_change: bool = False,
        # ---- architecture ----
        hidden_sizes: Tuple[int, ...] = (),
        hide_task_id: bool = False,
        # ---- BaseAgent / logging ----
        log_every: int = 1000,
        save_freq_epochs: int = 25,
        render_every: int = 0,
        post_burn_in_update_after: int = 0,
        experiment_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            logger=logger,
            run_ids=run_ids,
            agent_name="dqn",
            seed=seed,
            log_every=log_every,
            save_freq_epochs=save_freq_epochs,
            render_every=render_every,
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )

        # ---- DQN hyperparameters ----
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.double_dqn = double_dqn
        self.replay_size = int(replay_size)
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = int(batch_size)
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.n_updates = int(n_updates)
        self.post_burn_in_update_after = int(post_burn_in_update_after)

        # ---- buffer config ----
        self.buffer_mode = buffer_mode
        self.buffer_type = buffer_type
        self.per_scene_capacity = int(per_scene_capacity)
        self.flush_on = flush_on
        self.reset_buffer_on_task_change = reset_buffer_on_task_change
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_network_on_task_change = reset_network_on_task_change

        # ---- architecture ----
        self.hidden_sizes = hidden_sizes
        self.hide_task_id = hide_task_id

        # ---- networks + buffer + optimizer ----
        self._init_replay_buffer()

        policy_kwargs = self._policy_kwargs()
        self.q_network = models.MlpCritic(**policy_kwargs)
        self.target_q_network = models.MlpCritic(**policy_kwargs)
        self.target_q_network.set_weights(self.q_network.get_weights())

        self.optimizer = _keras.optimizers.legacy.Adam(learning_rate=lr)

        # Cached for ``on_episode_end`` so we can log epsilon per episode.
        self._last_epsilon: float = epsilon_start

        # Per-task compiled gradient step.  Recompiled in
        # :meth:`on_task_change` so the embedded ``current_task_idx``
        # is captured as a Python int constant — that lets the CL
        # method's trace-time branches (e.g. "skip the penalty on
        # task 0") bake into the per-task graph.
        self._learn_on_batch_fn = self._make_learn_on_batch_fn(0)

    # ==================================================================
    # Algorithm internals
    # ==================================================================

    def _policy_kwargs(self) -> Dict[str, Any]:
        return dict(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            num_tasks=self.num_tasks,
            hidden_sizes=self.hidden_sizes,
            hide_task_id=self.hide_task_id,
        )

    def _init_replay_buffer(self) -> None:
        if self.buffer_mode == "per_scene":
            from mariha.replay.buffers import PerSceneBufferPool

            buf_cls = (
                ReservoirReplayBuffer
                if self.buffer_type == BufferType.RESERVOIR
                else ReplayBuffer
            )
            self.replay_buffer = PerSceneBufferPool(
                obs_shape=self.obs_shape,
                per_scene_capacity=self.per_scene_capacity,
                num_tasks=self.num_tasks,
                buffer_factory=lambda shape, cap, nt: buf_cls(shape, cap, nt),
            )
        else:
            if self.buffer_type == BufferType.RESERVOIR:
                self.replay_buffer = ReservoirReplayBuffer(
                    self.obs_shape, self.replay_size, self.num_tasks
                )
            else:
                self.replay_buffer = ReplayBuffer(
                    self.obs_shape, self.replay_size, self.num_tasks
                )

    def _get_epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _eps_greedy(
        self, obs: np.ndarray, one_hot: np.ndarray, epsilon: float
    ) -> int:
        if np.random.random() < epsilon:
            # Sample directly from act_dim — works during burn-in when the
            # main env has been released (stable-retro single-emulator
            # constraint).
            return int(np.random.randint(0, self.act_dim))
        return self.get_action(obs, one_hot, deterministic=True)

    def _make_learn_on_batch_fn(self, current_task_idx: int):
        """Build a per-task compiled DQN gradient step.

        ``current_task_idx`` is captured in the closure as a Python
        int — same per-task recompile pattern as SAC's
        ``learn_on_batch`` and PPO's ``_gradient_step_fn``.  Recompiled
        in :meth:`on_task_change`.
        """

        @tf.function
        def learn_on_batch(
            obs,
            actions,
            rewards,
            next_obs,
            done,
            one_hot,
            episodic_batch=None,
        ):
            with tf.GradientTape() as tape:
                q_values = self.q_network(obs, one_hot)
                q_taken = tf.reduce_sum(
                    q_values * tf.one_hot(actions, self.act_dim), axis=-1
                )

                if self.double_dqn:
                    next_q_online = self.q_network(next_obs, one_hot)
                    next_actions = tf.argmax(
                        next_q_online, axis=-1, output_type=tf.int32
                    )
                    next_q_target = self.target_q_network(next_obs, one_hot)
                    next_q = tf.reduce_sum(
                        next_q_target * tf.one_hot(next_actions, self.act_dim),
                        axis=-1,
                    )
                else:
                    next_q_target = self.target_q_network(next_obs, one_hot)
                    next_q = tf.reduce_max(next_q_target, axis=-1)

                target = rewards + self.gamma * next_q * (1.0 - done)
                target = tf.stop_gradient(target)
                loss = tf.reduce_mean(tf.square(q_taken - target))

                # CL regularization (e.g. EWC quadratic anchor).
                # Returns ``tf.zeros([])`` on task 0 at trace time.
                if self.cl_method is not None:
                    loss = loss + self.cl_method.compute_loss_penalty(
                        self, task_idx=current_task_idx
                    )

            grads = tape.gradient(loss, self.q_network.trainable_variables)

            # Hand the gradients to the CL method (AGEM projection,
            # PackNet masking, DER additive distillation term, ...).
            grads_by_group: Dict[str, List[Optional[tf.Tensor]]] = {
                "q": list(grads)
            }
            if self.cl_method is not None:
                grads_by_group = self.cl_method.adjust_gradients(
                    self,
                    grads_by_group,
                    task_idx=current_task_idx,
                    episodic_batch=episodic_batch,
                )
            grads = grads_by_group["q"]

            self.optimizer.apply_gradients(
                zip(grads, self.q_network.trainable_variables)
            )

            # Polyak averaging for the target network
            for v, tv in zip(
                self.q_network.trainable_variables,
                self.target_q_network.trainable_variables,
            ):
                tv.assign(self.polyak * tv + (1.0 - self.polyak) * v)

            metrics = dict(
                loss=loss,
                q_mean=tf.reduce_mean(q_taken),
                abs_error=tf.abs(q_taken - target),
            )

            if self.cl_method is not None:
                self.cl_method.after_gradient_step(
                    self,
                    grads_by_group,
                    task_idx=current_task_idx,
                    metrics=metrics,
                )

            return metrics

        return learn_on_batch

    # ==================================================================
    # Agent-agnostic CL contract (Phase 4)
    # ==================================================================

    def get_named_parameter_groups(self) -> Dict[str, List[tf.Variable]]:
        """Named parameter groups for agent-agnostic CL methods.

        DQN has a single trainable network — the online Q-network.  The
        target network is tracked via Polyak averaging and is not
        regularized directly.  Returns the full ``trainable_variables``
        view so the single ``"q"`` group matches the optimizer view
        exactly — required for the ``adjust_gradients`` round-trip in
        :meth:`_learn_on_batch`.  EWC/MAS reduce naturally to this
        single group on this agent.
        """
        return {"q": list(self.q_network.trainable_variables)}

    def forward_for_importance(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Differentiable forward output per parameter group.

        Returns the online Q-values for the ``"q"`` group, used by
        EWC/MAS to compute Jacobians for importance estimation.
        """
        return {"q": self.q_network(obs, one_hot)}

    def distill_targets(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Distillation targets for DER/ClonEx-style CL methods.

        DQN distils Q-values rather than action logits.  The Phase 5
        ``DistillationMethod`` dispatches on the ``"q_values"`` key to
        apply the soft Q-value KL of Rusu et al. 2016 (*Policy
        Distillation*) — ``softmax(Q_old / T)`` vs
        ``softmax(Q_new / T)`` with a temperature hyperparameter.
        """
        return {"q_values": self.q_network(obs, one_hot)}

    def _compute_reference_loss(
        self, batch: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """DQN TD loss on ``batch`` with no CL penalty.

        Used by AGEM to compute reference gradients on episodic
        memory.  Mirrors the loss in :meth:`_make_learn_on_batch_fn`
        (Double DQN respected) but skips the CL penalty term and the
        Polyak target update.
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        one_hot = batch["one_hot"]

        q_values = self.q_network(obs, one_hot)
        q_taken = tf.reduce_sum(
            q_values * tf.one_hot(actions, self.act_dim), axis=-1
        )
        if self.double_dqn:
            next_q_online = self.q_network(next_obs, one_hot)
            next_actions = tf.argmax(
                next_q_online, axis=-1, output_type=tf.int32
            )
            next_q_target = self.target_q_network(next_obs, one_hot)
            next_q = tf.reduce_sum(
                next_q_target * tf.one_hot(next_actions, self.act_dim),
                axis=-1,
            )
        else:
            next_q_target = self.target_q_network(next_obs, one_hot)
            next_q = tf.reduce_max(next_q_target, axis=-1)
        target = tf.stop_gradient(
            rewards + self.gamma * next_q * (1.0 - done)
        )
        loss = tf.reduce_mean(tf.square(q_taken - target))
        return {"q": loss}

    # ==================================================================
    # BenchmarkAgent contract
    # ==================================================================

    @tf.function
    def _get_action_tf(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> tf.Tensor:
        q_values = self.q_network(obs, one_hot)
        return tf.cast(tf.argmax(q_values[0]), tf.int32)

    def get_action(
        self,
        obs: np.ndarray,
        task_one_hot: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
        one_hot_t = tf.expand_dims(tf.cast(task_one_hot, tf.float32), 0)
        return int(self._get_action_tf(obs_t, one_hot_t).numpy())

    # ==================================================================
    # Required BaseAgent callbacks
    # ==================================================================

    def select_action(
        self,
        *,
        obs: np.ndarray,
        one_hot: np.ndarray,
        global_step: int,
        task_step: int,
        current_task_idx: int,
    ) -> Tuple[int, Dict[str, Any]]:
        epsilon = self._get_epsilon(global_step)
        self._last_epsilon = epsilon
        action = self._eps_greedy(obs, one_hot, epsilon)
        return action, {}

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
        # Truncated episodes (timeouts) must store ``done=False`` so the
        # bootstrap target keeps a future term — only true terminations
        # zero out the next-state value.
        done_to_store = terminated
        if self.buffer_mode == "per_scene":
            self.replay_buffer.store(
                scene_id, obs, action, reward, next_obs, done_to_store, one_hot
            )
        else:
            self.replay_buffer.store(
                obs, action, reward, next_obs, done_to_store, one_hot
            )

    def should_update(self, global_step: int) -> bool:
        return (
            global_step >= self.update_after
            and global_step % self.update_every == 0
            and self.replay_buffer.size >= self.batch_size
        )

    def update_step(
        self, *, global_step: int, current_task_idx: int
    ) -> None:
        for _ in range(self.n_updates):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            episodic_batch = (
                self.cl_method.get_episodic_batch(
                    self, task_idx=current_task_idx
                )
                if self.cl_method is not None
                else None
            )
            results = self._learn_on_batch_fn(
                batch["obs"],
                batch["actions"],
                batch["rewards"],
                batch["next_obs"],
                batch["done"],
                batch["one_hot"],
                episodic_batch,
            )
            self.logger.store(
                {
                    "train/loss_q": float(results["loss"]),
                    "train/q_mean": float(results["q_mean"]),
                }
            )

    def save_weights(self, directory: Path) -> None:
        directory = Path(directory)
        self.q_network.save_weights(str(directory / "q_network"))
        self.target_q_network.save_weights(
            str(directory / "target_q_network")
        )

    def load_weights(self, directory: Path) -> None:
        directory = Path(directory)
        self.q_network.load_weights(str(directory / "q_network"))
        self.target_q_network.load_weights(
            str(directory / "target_q_network")
        )

    # ==================================================================
    # Optional BaseAgent hooks
    # ==================================================================

    def on_task_start(
        self, current_task_idx: int, run_id: str = ""
    ) -> None:
        """DQN's task-start log line + forward to attached CL method."""
        _run = run_id or (
            self.run_ids[current_task_idx]
            if current_task_idx < len(self.run_ids)
            else "?"
        )
        self.logger.log(
            f"Task start: idx={current_task_idx}  run={_run}",
            color="white",
        )
        super().on_task_start(current_task_idx, _run)

    def on_task_end(self, current_task_idx: int) -> None:
        """DQN's task-end log line + forward to attached CL method."""
        self.logger.log(f"Task end:   idx={current_task_idx}", color="white")
        super().on_task_end(current_task_idx)

    def on_task_change(self, new_task_idx: int, new_run_id: str) -> None:
        if self.reset_buffer_on_task_change and self.buffer_mode != "per_scene":
            self._init_replay_buffer()
        if self.reset_optimizer_on_task_change:
            from mariha.utils.running import reset_optimizer

            reset_optimizer(self.optimizer)
        if self.reset_network_on_task_change:
            from mariha.utils.running import reset_weights

            reset_weights(
                self.q_network, models.MlpCritic, self._policy_kwargs()
            )
            self.target_q_network.set_weights(self.q_network.get_weights())

        # Recompile so the embedded ``current_task_idx`` constant
        # matches the new task — see ``_make_learn_on_batch_fn``.
        self._learn_on_batch_fn = self._make_learn_on_batch_fn(new_task_idx)

    def handle_session_boundary(self, current_task_idx: int) -> None:
        """Flush per-scene buffers with gradient updates at session boundaries."""
        if self.buffer_mode != "per_scene" or self.flush_on != "session":
            return

        from mariha.replay.buffers import PerSceneBufferPool

        pool: PerSceneBufferPool = self.replay_buffer  # type: ignore[assignment]
        total = pool.total_size
        if total >= self.batch_size:
            n_updates = total // self.batch_size
            self.logger.log(
                f"[session-flush] DQN: {total} transitions — "
                f"{n_updates} updates.",
                color="cyan",
            )
            for _ in range(n_updates):
                batch = pool.sample_batch(self.batch_size)
                episodic_batch = (
                    self.cl_method.get_episodic_batch(
                        self, task_idx=current_task_idx
                    )
                    if self.cl_method is not None
                    else None
                )
                self._learn_on_batch_fn(
                    batch["obs"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_obs"],
                    batch["done"],
                    batch["one_hot"],
                    episodic_batch,
                )

        flushed = pool.flush_all()
        self.logger.log(
            f"[session-flush] Flushed {flushed} transitions.", color="cyan"
        )

    def on_episode_end(
        self,
        *,
        episode_return: float,
        episode_len: int,
        total_episodes: int,
    ) -> None:
        super().on_episode_end(
            episode_return=episode_return,
            episode_len=episode_len,
            total_episodes=total_episodes,
        )
        buf_pct = (
            self.replay_buffer.size / max(self.replay_buffer.max_size, 1) * 100
        )
        self.logger.store(
            {
                "train/epsilon": float(self._last_epsilon),
                "buffer_fill_pct": float(buf_pct),
            }
        )

    def get_log_tabular_keys(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("train/epsilon", {"average_only": True}),
            ("buffer_fill_pct", {"average_only": True}),
            ("train/loss_q", {"average_only": True}),
            ("train/q_mean", {"average_only": True}),
        ]

    # ==================================================================
    # Burn-in
    # ==================================================================

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        run_burn_in(self, burn_in_spec, num_steps)

    def on_burn_in_end(self) -> None:
        """Reset the replay buffer and promote ``post_burn_in_update_after``.

        After burn-in, the buffer is filled with single-scene data which
        would heavily bias the first updates of real training.  Wiping
        it forces the agent to start fresh.  ``update_after`` is also
        promoted from its training default to the post-burn-in value
        (typically 0, since the buffer just got reset).
        """
        self._init_replay_buffer()
        self.update_after = self.post_burn_in_update_after
        self.logger.log(
            f"[burn-in] DQN: buffer reset, update_after={self.update_after}",
            color="cyan",
        )

    # ==================================================================
    # Config interface
    # ==================================================================

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        from mariha.replay.buffers import BufferType
        from mariha.utils.running import sci2int, str2bool

        parser.add_argument("--epsilon_start", type=float, default=1.0)
        parser.add_argument("--epsilon_end", type=float, default=0.01)
        parser.add_argument("--epsilon_decay_steps", type=sci2int, default=50_000)
        parser.add_argument("--double_dqn", type=str2bool, default=True)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--polyak", type=float, default=0.995)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--replay_size", type=sci2int, default=100_000)
        parser.add_argument("--update_after", type=sci2int, default=5_000)
        parser.add_argument("--update_every", type=int, default=4)
        parser.add_argument("--n_updates", type=int, default=1)
        parser.add_argument(
            "--hidden_sizes", type=int, nargs="*", default=[]
        )
        parser.add_argument("--hide_task_id", type=str2bool, default=False)
        parser.add_argument(
            "--buffer_type",
            type=str,
            default="fifo",
            choices=[bt.value for bt in BufferType],
        )
        parser.add_argument(
            "--reset_buffer_on_task_change", type=str2bool, default=True
        )
        parser.add_argument(
            "--reset_optimizer_on_task_change", type=str2bool, default=False
        )
        parser.add_argument(
            "--reset_network_on_task_change", type=str2bool, default=False
        )
        parser.add_argument("--log_every", type=sci2int, default=1000)
        parser.add_argument("--save_freq_epochs", type=int, default=25)
        parser.add_argument(
            "--render_every",
            type=int,
            default=0,
            help="Play a greedy episode every N training episodes (0=disabled).",
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, env, logger, run_ids: list
    ) -> "DQN":
        from mariha.utils.running import get_readable_timestamp

        experiment_dir = Path(getattr(args, "experiment_dir", "experiments"))
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
            epsilon_start=getattr(args, "epsilon_start", 1.0),
            epsilon_end=getattr(args, "epsilon_end", 0.01),
            epsilon_decay_steps=getattr(args, "epsilon_decay_steps", 50_000),
            double_dqn=getattr(args, "double_dqn", True),
            replay_size=getattr(args, "replay_size", 100_000),
            gamma=getattr(args, "gamma", 0.99),
            polyak=getattr(args, "polyak", 0.995),
            lr=getattr(args, "lr", 1e-4),
            batch_size=getattr(args, "batch_size", 32),
            update_after=getattr(args, "update_after", 5_000),
            update_every=getattr(args, "update_every", 4),
            n_updates=getattr(args, "n_updates", 1),
            buffer_mode=getattr(args, "buffer_mode", "single"),
            buffer_type=BufferType(getattr(args, "buffer_type", "fifo")),
            per_scene_capacity=getattr(args, "per_scene_capacity", 1000),
            flush_on=getattr(args, "flush_on", "session"),
            reset_buffer_on_task_change=getattr(
                args, "reset_buffer_on_task_change", True
            ),
            reset_optimizer_on_task_change=getattr(
                args, "reset_optimizer_on_task_change", False
            ),
            reset_network_on_task_change=getattr(
                args, "reset_network_on_task_change", False
            ),
            hidden_sizes=tuple(getattr(args, "hidden_sizes", [])),
            hide_task_id=getattr(args, "hide_task_id", False),
            log_every=getattr(args, "log_every", 1000),
            save_freq_epochs=getattr(args, "save_freq_epochs", 25),
            render_every=getattr(args, "render_every", 0),
            post_burn_in_update_after=getattr(
                args, "post_burn_in_update_after", 0
            ),
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )
