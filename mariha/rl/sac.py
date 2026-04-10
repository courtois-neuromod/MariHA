"""Soft Actor-Critic for the MariHA continual-learning benchmark.

``SAC`` is the base training class for all methods (vanilla SAC and all CL
baselines).  CL methods subclass it and override:

- ``adjust_gradients()`` — modify gradients before the optimizer step.
- ``get_auxiliary_loss()`` — add a regularisation term to the loss.
- ``on_task_start()`` / ``on_task_end()`` — lifecycle hooks.
- ``get_episodic_batch()`` — supply a replay batch from past tasks.

The training loop is **episode-driven**: it consumes one ``EpisodeSpec`` per
episode from ``ContinualLearningEnv`` until the curriculum is exhausted.
This replaces COOM's fixed ``steps_per_env * num_tasks`` budget.

Key differences from COOM:
- No ``test_envs`` in the base loop (evaluation is a separate phase).
- ``max_episode_len`` is set per episode by ``SceneEnv`` (truncation via
  ``EpisodeSpec.max_steps``), not by a global game-timeout query.
- ``seq_idx`` (passed to ``learn_on_batch``) is the index of the current
  scene in ``scene_ids`` (0–312 for the full benchmark).
- ``one_hot_vec`` is read from ``info['task_one_hot']`` after each reset.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf

# TF 2.17+ ships Keras 3 by default; tf_keras provides Keras 2 compatibility.
# We use tf_keras when available so that all keras.* imports (layers, optimizers,
# etc.) come from the same Keras 2 build that tensorflow-probability expects.
try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

from tensorflow_probability.python.distributions import Categorical

from mariha.benchmark.agent import BenchmarkAgent
from mariha.replay.buffers import (
    BufferType,
    EpisodicMemory,
    PrioritizedExperienceReplay,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReservoirReplayBuffer,
)
from mariha.rl import models
from mariha.utils.logging import EpochLogger
from mariha.utils.running import (
    create_one_hot_vec,
    reset_optimizer,
    reset_weights,
    set_seed,
)


class SAC(BenchmarkAgent):
    """Soft Actor-Critic for continual learning on the MariHA benchmark.

    After construction, call ``run()`` to begin training.

    Args:
        env: A ``ContinualLearningEnv`` instance (or any compatible env).
        logger: An ``EpochLogger`` for recording training statistics.
        scene_ids: Ordered list of all scene IDs in the benchmark.  Defines
            the one-hot dimension and the ``seq_idx`` mapping.
        agent: Agent name string (used for checkpoint paths).
        actor_cl: Actor class to instantiate.
        critic_cl: Critic class to instantiate.
        policy_kwargs: Additional keyword arguments forwarded to both the
            actor and critic constructors.
        seed: Global random seed.
        log_every: Number of environment steps between logging epochs.
        replay_size: Replay buffer capacity (number of transitions).
        gamma: Discount factor.
        polyak: Polyak averaging coefficient for target networks.
            Target updated as ``τ·target + (1−τ)·online``.
        lr: Initial learning rate.
        lr_decay: Learning rate schedule type: ``None``, ``'exponential'``,
            or ``'linear'``.
        lr_decay_rate: Decay factor (for exponential) or end-LR ratio
            (for linear).
        lr_decay_steps: Number of optimizer steps over which to decay.
        alpha: Entropy regularisation coefficient.  Use ``'auto'`` for
            automatic tuning (recommended).
        batch_size: Mini-batch size for gradient updates.
        start_steps: Number of steps with random action selection before
            the policy is used (per task if ``agent_policy_exploration=False``).
        update_after: Minimum transitions in the buffer before updates start.
        update_every: Steps between consecutive update rounds.
        n_updates: Number of gradient steps per update round.
        save_freq_epochs: Checkpointing frequency (in logging epochs).
        reset_buffer_on_task_change: Clear the replay buffer at task switches.
        buffer_type: Which replay buffer implementation to use.
        reset_optimizer_on_task_change: Reset Adam slot variables at task switches.
        reset_actor_on_task_change: Re-initialise actor weights at task switches.
        reset_critic_on_task_change: Re-initialise critic weights at task switches.
        clipnorm: Global gradient-norm clipping threshold (``None`` = disabled).
        agent_policy_exploration: If ``True``, random exploration is only used
            during the first task; subsequent tasks start with the trained policy.
        experiment_dir: Root directory for checkpoints and logs.
        timestamp: Run timestamp string (for checkpoint naming).
    """

    def __init__(
        self,
        env,
        logger: EpochLogger,
        scene_ids: List[str],
        agent: Optional[str] = None,
        actor_cl: Type = models.MlpActor,
        critic_cl: Type = models.MlpCritic,
        policy_kwargs: Optional[Dict] = None,
        seed: int = 0,
        log_every: int = 1000,
        replay_size: int = 100_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-3,
        lr_decay: Optional[str] = None,
        lr_decay_rate: float = 0.1,
        lr_decay_steps: Optional[int] = None,
        alpha: Union[float, str] = "auto",
        batch_size: int = 128,
        start_steps: int = 10_000,
        update_after: int = 5_000,
        update_every: int = 50,
        n_updates: int = 50,
        save_freq_epochs: int = 25,
        reset_buffer_on_task_change: bool = True,
        buffer_type: BufferType = BufferType.FIFO,
        reset_optimizer_on_task_change: bool = False,
        reset_actor_on_task_change: bool = False,
        reset_critic_on_task_change: bool = False,
        clipnorm: Optional[float] = None,
        agent_policy_exploration: bool = False,
        render_every: int = 0,
        post_burn_in_update_after: int = 0,
        buffer_mode: str = "single",
        per_scene_capacity: int = 1000,
        flush_on: str = "session",
        cl_hook_min_transitions: int = 500,
        experiment_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        set_seed(seed, env=env)

        if policy_kwargs is None:
            policy_kwargs = {}

        self.env = env
        self.scene_ids = scene_ids
        self.num_tasks = len(scene_ids)
        self.logger = logger
        self.agent = agent
        self.critic_cl = critic_cl
        self.policy_kwargs = policy_kwargs
        self.replay_size = int(replay_size)
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = int(start_steps)
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.n_updates = int(n_updates)
        self.log_every = int(log_every)
        self.save_freq_epochs = save_freq_epochs
        self.reset_buffer_on_task_change = reset_buffer_on_task_change
        self.buffer_type = buffer_type
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_actor_on_task_change = reset_actor_on_task_change
        self.reset_critic_on_task_change = reset_critic_on_task_change
        self.clipnorm = clipnorm
        self.agent_policy_exploration = agent_policy_exploration
        self.render_every = int(render_every)
        self.post_burn_in_update_after = int(post_burn_in_update_after)
        self.buffer_mode = buffer_mode
        self.per_scene_capacity = per_scene_capacity
        self.flush_on = flush_on
        self.cl_hook_min_transitions = cl_hook_min_transitions
        self.experiment_dir = experiment_dir or Path("experiments")
        self.timestamp = timestamp

        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n
        logger.log(f"Observation shape: {self.obs_shape}", color="blue")
        logger.log(f"Action dim:        {self.act_dim}", color="blue")
        logger.log(f"Num tasks:         {self.num_tasks}", color="blue")

        # Share spaces + num_tasks with model constructors.
        policy_kwargs["state_space"] = env.observation_space
        policy_kwargs["action_space"] = env.action_space
        policy_kwargs["num_tasks"] = self.num_tasks

        # Replay buffer
        self._init_replay_buffer()

        # Networks
        self.actor_cl = actor_cl
        self.actor_kwargs = policy_kwargs
        self.actor = actor_cl(**policy_kwargs)

        self.critic1 = critic_cl(**policy_kwargs)
        self.target_critic1 = critic_cl(**policy_kwargs)
        self.target_critic1.set_weights(self.critic1.get_weights())

        self.critic2 = critic_cl(**policy_kwargs)
        self.target_critic2 = critic_cl(**policy_kwargs)
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.critic_variables = (
            self.critic1.trainable_variables + self.critic2.trainable_variables
        )
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )

        # Learning rate schedule
        if lr_decay is not None and lr_decay_steps is None:
            # Heuristic: enough steps for one pass through the curriculum
            lr_decay_steps = 500_000
        if lr_decay == "exponential":
            lr = _keras.optimizers.schedules.ExponentialDecay(lr, lr_decay_steps, lr_decay_rate)
        elif lr_decay == "linear":
            lr = _keras.optimizers.schedules.PolynomialDecay(
                lr, lr_decay_steps, lr * lr_decay_rate, power=1.0
            )

        self.optimizer = _keras.optimizers.legacy.Adam(learning_rate=lr)

        # Automatic entropy tuning
        self.auto_alpha = alpha == "auto"
        if self.auto_alpha:
            # One log-alpha per task (scene).
            self.all_log_alpha = tf.Variable(
                np.zeros((self.num_tasks, 1), dtype=np.float32), trainable=True
            )
            # Target entropy: fraction of maximum discrete entropy.
            self.target_entropy = 0.98 * np.log(self.act_dim)

    # ------------------------------------------------------------------
    # Replay buffer initialisation
    # ------------------------------------------------------------------

    def _init_replay_buffer(self) -> None:
        """(Re-)create the replay buffer according to ``self.buffer_type``."""
        if self.buffer_mode == "per_scene":
            from mariha.replay.buffers import PerSceneBufferPool
            buffer_cls = self._get_buffer_cls()
            self.replay_buffer = PerSceneBufferPool(
                obs_shape=self.obs_shape,
                per_scene_capacity=self.per_scene_capacity,
                num_tasks=self.num_tasks,
                buffer_cls=buffer_cls,
            )
        else:
            kwargs = dict(
                obs_shape=self.obs_shape,
                size=self.replay_size,
                num_tasks=self.num_tasks,
            )
            if self.buffer_type == BufferType.FIFO:
                self.replay_buffer = ReplayBuffer(**kwargs)
            elif self.buffer_type == BufferType.RESERVOIR:
                self.replay_buffer = ReservoirReplayBuffer(**kwargs)
            elif self.buffer_type == BufferType.PRIORITY:
                self.replay_buffer = PrioritizedReplayBuffer(**kwargs)
            elif self.buffer_type == BufferType.PER:
                self.replay_buffer = PrioritizedExperienceReplay(**kwargs)
            else:
                raise ValueError(f"Unknown buffer type: {self.buffer_type}")

    def _get_buffer_cls(self) -> type:
        """Return the buffer class for the configured buffer type."""
        if self.buffer_type == BufferType.FIFO:
            return ReplayBuffer
        if self.buffer_type == BufferType.RESERVOIR:
            return ReservoirReplayBuffer
        # PER + per_scene not supported yet.
        raise ValueError(
            f"Buffer type '{self.buffer_type.value}' is not supported with "
            "buffer_mode='per_scene'.  Use 'fifo' or 'reservoir'."
        )

    # ------------------------------------------------------------------
    # CL extension points (overridden by subclasses)
    # ------------------------------------------------------------------

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: Optional[List[tf.Tensor]],
        current_task_idx: int,
        metrics: Dict,
        episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], Optional[List[tf.Tensor]]]:
        """Hook for CL methods to modify gradients. Base: identity."""
        return actor_gradients, critic_gradients, alpha_gradient

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        """Hook for CL regularisation losses. Base: zero."""
        return tf.constant(0.0)

    def on_test_start(self, seq_idx: Union[tf.Tensor, int]) -> None:
        pass

    def on_test_end(self, seq_idx: Union[tf.Tensor, int]) -> None:
        pass

    def on_task_start(self, current_task_idx: int, scene_id: str = "") -> None:
        _scene = scene_id or (
            self.scene_ids[current_task_idx] if current_task_idx < len(self.scene_ids) else "?"
        )
        self.logger.log(f"Task start: idx={current_task_idx}  scene={_scene}", color="white")

    def on_task_end(self, current_task_idx: int) -> None:
        self.logger.log(f"Task end:   idx={current_task_idx}", color="white")

    def get_episodic_batch(
        self, current_task_idx: int
    ) -> Optional[Dict[str, tf.Tensor]]:
        """Hook for AGEM / DER++ style episodic memory. Base: None."""
        return None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_log_alpha(self, one_hot: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(
            tf.linalg.matmul(
                tf.expand_dims(tf.convert_to_tensor(one_hot), 1), self.all_log_alpha
            )
        )

    @tf.function
    def _get_action_tf(
        self,
        obs: tf.Tensor,
        one_hot_task_id: tf.Tensor,
        deterministic: tf.Tensor = tf.constant(False),
    ) -> tf.Tensor:
        """TF-compiled action selection from the current policy."""
        logits = self.actor(
            tf.expand_dims(obs, 0), tf.expand_dims(one_hot_task_id, 0)
        )
        dist = Categorical(logits=logits)
        return (
            tf.math.argmax(logits, axis=-1, output_type=tf.int32)
            if deterministic
            else dist.sample()
        )

    def get_action_numpy(
        self,
        obs: np.ndarray,
        one_hot: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Convenience wrapper: return a numpy int action."""
        action = self._get_action_tf(
            tf.convert_to_tensor(obs),
            tf.convert_to_tensor(one_hot, dtype=tf.float32),
            tf.constant(deterministic),
        )
        return int(action.numpy()[0])

    def get_action(
        self,
        obs: np.ndarray,
        task_one_hot: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """BenchmarkAgent interface: select an action (numpy in, int out)."""
        return self.get_action_numpy(obs, task_one_hot, deterministic)

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def get_learn_on_batch(self, current_task_idx: int) -> Callable:
        """Return a compiled ``learn_on_batch`` function for ``current_task_idx``."""

        @tf.function
        def learn_on_batch(
            seq_idx: tf.Tensor,
            batch: Dict[str, tf.Tensor],
            episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
        ) -> Dict:
            gradients, metrics = self.get_gradients(seq_idx, **batch)
            gradients = self.adjust_gradients(
                *gradients,
                current_task_idx=current_task_idx,
                metrics=metrics,
                episodic_batch=episodic_batch,
            )
            if self.clipnorm is not None:
                actor_grads, critic_grads, alpha_grad = gradients
                gradients = (
                    tf.clip_by_global_norm(actor_grads, self.clipnorm)[0],
                    tf.clip_by_global_norm(critic_grads, self.clipnorm)[0],
                    tf.clip_by_norm(alpha_grad, self.clipnorm)
                    if alpha_grad is not None
                    else None,
                )
            self.apply_update(*gradients)
            return metrics

        return learn_on_batch

    def get_gradients(
        self,
        seq_idx: tf.Tensor,
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
        one_hot: tf.Tensor,
        **kwargs,
    ) -> Tuple[Tuple, Dict]:
        """Compute SAC actor, critic, and alpha gradients."""
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(one_hot)
            else:
                log_alpha = tf.math.log(tf.cast(self.alpha, tf.float32))
            log_alpha_exp = tf.math.exp(log_alpha)

            logits = self.actor(obs, one_hot)
            dist = Categorical(logits=logits)
            entropy = dist.entropy()

            logits_next = self.actor(next_obs, one_hot)
            dist_next = Categorical(logits=logits_next)
            entropy_next = dist_next.entropy()

            q1 = self.critic1(obs, one_hot)
            q2 = self.critic2(obs, one_hot)

            q1_vals = tf.gather(q1, actions, axis=1, batch_dims=1)
            q2_vals = tf.gather(q2, actions, axis=1, batch_dims=1)

            target_q1 = self.target_critic1(next_obs, one_hot)
            target_q2 = self.target_critic2(next_obs, one_hot)

            min_q = dist.probs_parameter() * tf.stop_gradient(tf.minimum(q1, q2))
            min_target_q = dist_next.probs_parameter() * tf.minimum(target_q1, target_q2)

            q_backup = tf.stop_gradient(
                rewards
                + self.gamma
                * (1 - done)
                * (tf.math.reduce_sum(min_target_q, axis=-1) - log_alpha_exp * entropy_next)
            )

            abs_error = tf.stop_gradient(
                tf.math.minimum(tf.abs(q_backup - q1_vals), tf.abs(q_backup - q2_vals))
            )

            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_vals) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_vals) ** 2)
            value_loss = q1_loss + q2_loss

            actor_loss = -tf.reduce_mean(
                log_alpha_exp * entropy + tf.reduce_sum(min_q, axis=-1)
            )

            if self.auto_alpha:
                log_prob = tf.stop_gradient(entropy) + self.target_entropy
                alpha_loss = -tf.reduce_mean(log_alpha * log_prob)

            auxiliary_loss = self.get_auxiliary_loss(seq_idx)
            actor_loss += auxiliary_loss
            value_loss += auxiliary_loss

        actor_gradients = g.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        alpha_gradient = (
            g.gradient(alpha_loss, self.all_log_alpha) if self.auto_alpha else None
        )
        del g

        metrics = dict(
            pi_loss=actor_loss,
            q1_loss=q1_loss,
            q2_loss=q2_loss,
            q1=q1_vals,
            q2=q2_vals,
            entropy=entropy,
            reg_loss=auxiliary_loss,
            kl_loss=0,
            agem_violation=0,
            abs_error=abs_error,
        )
        return (actor_gradients, critic_gradients, alpha_gradient), metrics

    def apply_update(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: Optional[tf.Tensor],
    ) -> None:
        """Apply gradients and update target networks via Polyak averaging."""
        self.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )
        self.optimizer.apply_gradients(
            zip(critic_gradients, self.critic_variables)
        )
        if self.auto_alpha and alpha_gradient is not None:
            self.optimizer.apply_gradients([(alpha_gradient, self.all_log_alpha)])

        for v, tv in zip(self.critic1.trainable_variables, self.target_critic1.trainable_variables):
            tv.assign(self.polyak * tv + (1 - self.polyak) * v)
        for v, tv in zip(self.critic2.trainable_variables, self.target_critic2.trainable_variables):
            tv.assign(self.polyak * tv + (1 - self.polyak) * v)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _handle_task_change(self, current_task_idx: int) -> None:
        """Reset buffers/weights/optimizer as configured, then recompile update."""
        self.on_task_start(current_task_idx)

        if self.reset_buffer_on_task_change and self.buffer_mode != "per_scene":
            self._init_replay_buffer()

        if self.reset_actor_on_task_change:
            reset_weights(self.actor, self.actor_cl, self.actor_kwargs)

        if self.reset_critic_on_task_change:
            reset_weights(self.critic1, self.critic_cl, self.policy_kwargs)
            self.target_critic1.set_weights(self.critic1.get_weights())
            reset_weights(self.critic2, self.critic_cl, self.policy_kwargs)
            self.target_critic2.set_weights(self.critic2.get_weights())

        if self.reset_optimizer_on_task_change:
            self.logger.log("Resetting optimizer.", color="cyan")
            reset_optimizer(self.optimizer)

        # Recompile so TensorFlow picks up any trainability changes.
        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )

    # ------------------------------------------------------------------
    # Session boundary handling (per-scene buffer mode)
    # ------------------------------------------------------------------

    def _handle_session_boundary(self, current_task_idx: int) -> None:
        """Flush per-scene buffers and run end-of-session gradient updates."""
        from mariha.replay.buffers import PerSceneBufferPool

        pool: PerSceneBufferPool = self.replay_buffer  # type: ignore[assignment]
        total = pool.total_size
        if total >= self.batch_size:
            n_updates = total // self.batch_size
            self.logger.log(
                f"[session-flush] {total} transitions across "
                f"{len(pool.active_scene_ids)} scenes — "
                f"{n_updates} gradient updates.",
                color="cyan",
            )
            t_start = time.time()
            for _ in range(n_updates):
                batch = pool.sample_batch(self.batch_size)
                episodic_batch = self.get_episodic_batch(current_task_idx)
                results = self.learn_on_batch(
                    tf.constant(current_task_idx), batch, episodic_batch
                )
                self._log_after_update(results)
            self.logger.log(
                f"[session-flush] Done in {time.time() - t_start:.2f}s.",
                color="cyan",
            )
        else:
            self.logger.log(
                f"[session-flush] Only {total} transitions — skipping updates.",
                color="yellow",
            )

        # Fire CL hooks with a full session's worth of data.
        self.on_task_end(current_task_idx)
        flushed = pool.flush_all()
        self.logger.log(
            f"[session-flush] Flushed {flushed} transitions.", color="cyan"
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_after_update(self, results: Dict) -> None:
        self.logger.store(
            {
                "train/q1_vals": results["q1"],
                "train/q2_vals": results["q2"],
                "train/entropy": results["entropy"],
                "train/loss_kl": results["kl_loss"],
                "train/loss_pi": results["pi_loss"],
                "train/loss_q1": results["q1_loss"],
                "train/loss_q2": results["q2_loss"],
                "train/loss_reg": results["reg_loss"],
                "train/agem_violation": results["agem_violation"],
            }
        )
        if self.auto_alpha:
            for task_idx in range(self.num_tasks):
                self.logger.store(
                    {f"train/alpha/{task_idx}": float(tf.math.exp(self.all_log_alpha[task_idx][0]))}
                )

    def _log_after_epoch(self, epoch: int, global_timestep: int, action_counts: Dict) -> None:
        self.logger.log_tabular("epoch", epoch)
        self.logger.log_tabular("total_env_steps", global_timestep + 1)
        self.logger.log_tabular("train/return", with_min_and_max=True)
        self.logger.log_tabular("train/ep_length", average_only=True)
        self.logger.log_tabular("train/episodes", average_only=True)
        self.logger.log_tabular("buffer_fill_pct", average_only=True)
        self.logger.log_tabular("train/loss_pi", average_only=True)
        self.logger.log_tabular("train/loss_q1", average_only=True)
        self.logger.log_tabular("train/loss_q2", average_only=True)
        self.logger.log_tabular("train/loss_kl", average_only=True)
        self.logger.log_tabular("train/loss_reg", average_only=True)
        from mariha.env.wrappers.action import ACTION_NAMES
        for i, count in action_counts.items():
            name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else str(i)
            self.logger.log_tabular(f"train/actions/{name}", count)
        self.logger.log_tabular("walltime", time.time() - self.start_time)
        self.logger.dump_tabular()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_model(self, current_task_idx: int) -> None:
        """Save actor and critic weights to disk."""
        agent_name = self.agent or "sac"
        ts = self.timestamp or ""
        model_dir = str(self.experiment_dir / "checkpoints" / agent_name / f"{ts}_task{current_task_idx}")
        self.logger.log(f"Saving model to {model_dir}", color="crimson")
        os.makedirs(model_dir, exist_ok=True)
        self.actor.save_weights(os.path.join(model_dir, "actor"))
        self.critic1.save_weights(os.path.join(model_dir, "critic1"))
        self.target_critic1.save_weights(os.path.join(model_dir, "target_critic1"))
        self.critic2.save_weights(os.path.join(model_dir, "critic2"))
        self.target_critic2.save_weights(os.path.join(model_dir, "target_critic2"))

    def load_model(self, model_dir: str) -> None:
        """Load actor and critic weights from a checkpoint directory."""
        self.actor.load_weights(os.path.join(model_dir, "actor"))
        self.critic1.load_weights(os.path.join(model_dir, "critic1"))
        self.target_critic1.load_weights(os.path.join(model_dir, "target_critic1"))
        self.critic2.load_weights(os.path.join(model_dir, "critic2"))
        self.target_critic2.load_weights(os.path.join(model_dir, "target_critic2"))

    # BenchmarkAgent checkpoint interface (delegates to save_model/load_model)

    def save_checkpoint(self, directory: Path) -> None:
        """BenchmarkAgent interface: save weights to ``directory``."""
        self.save_model(str(directory))

    def load_checkpoint(self, directory: Path) -> None:
        """BenchmarkAgent interface: load weights from ``directory``."""
        self.load_model(str(directory))

    # ------------------------------------------------------------------
    # Burn-in
    # ------------------------------------------------------------------

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        """Pre-train on a single scene to warm up the replay buffer and policy.

        Repeatedly resets and plays ``burn_in_spec`` until *num_steps*
        transitions are collected.  Standard SAC updates run during burn-in.
        After completion the replay buffer is flushed.

        No CL hooks fire during burn-in and the data does not count toward
        CL metrics.

        Args:
            burn_in_spec: ``EpisodeSpec`` for the burn-in scene.
            num_steps: Total environment steps to collect.
        """
        from mariha.env.continual import make_scene_env
        from mariha.env.scenario_gen import load_metadata
        from mariha.env.base import SCENARIOS_DIR

        scene_id = burn_in_spec.scene_id
        self.logger.log(
            f"[burn-in] Starting burn-in on '{scene_id}' for {num_steps} steps.",
            color="cyan",
        )

        # Build a dedicated env for burn-in. The caller (run_cl.py) is
        # responsible for releasing/reacquiring the main env around this
        # call — stable-retro allows only one emulator per process.
        scene_meta = load_metadata(SCENARIOS_DIR)
        exit_point = scene_meta[scene_id]["exit_point"]
        burn_env = make_scene_env(
            scene_id=scene_id,
            exit_point=exit_point,
            scene_ids=self.scene_ids,
            render_mode=None,
        )

        # Compile the update function for the burn-in scene.
        task_idx = (
            self.scene_ids.index(scene_id) if scene_id in self.scene_ids else 0
        )
        self.learn_on_batch = self.get_learn_on_batch(task_idx)

        episode_return = 0.0
        episode_len = 0
        episodes = 0
        step = 0
        t_start = time.time()

        try:
            obs, info = burn_env.reset(episode_spec=burn_in_spec)
            one_hot_vec = info["task_one_hot"]

            while step < num_steps:
                # Action selection: random until start_steps, then policy.
                if step < self.start_steps:
                    action = burn_env.action_space.sample()
                else:
                    action = self.get_action_numpy(obs, one_hot_vec)

                next_obs, reward, terminated, truncated, _info = burn_env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_len += 1

                done_to_store = terminated  # truncation != terminal for replay
                self.replay_buffer.store(
                    obs, action, reward, next_obs, done_to_store, one_hot_vec
                )
                obs = next_obs

                if done:
                    episodes += 1
                    self.logger.log(
                        f"[burn-in] Ep {episodes:4d} | return={episode_return:7.2f} | "
                        f"len={episode_len:4d} | step={step}/{num_steps}"
                    )
                    episode_return = 0.0
                    episode_len = 0
                    obs, info = burn_env.reset(episode_spec=burn_in_spec)
                    one_hot_vec = info["task_one_hot"]

                # Policy update.
                if (
                    step >= self.update_after
                    and step % self.update_every == 0
                    and self.replay_buffer.size >= self.batch_size
                ):
                    for _ in range(self.n_updates):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        results = self.learn_on_batch(
                            tf.constant(task_idx), batch, None
                        )
                        self._log_after_update(results)

                step += 1
        finally:
            burn_env.close()

        elapsed = time.time() - t_start
        self.logger.log(
            f"[burn-in] Complete — {step} steps, {episodes} episodes in {elapsed:.1f}s.",
            color="cyan",
        )

        # Flush the buffer so burn-in data doesn't contaminate the curriculum.
        self._init_replay_buffer()

        # Set post-burn-in update_after.
        self.update_after = self.post_burn_in_update_after
        self.logger.log(
            f"[burn-in] update_after set to {self.update_after} for curriculum.",
            color="cyan",
        )

    # ------------------------------------------------------------------
    # BenchmarkAgent config interface
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add SAC and CL-method hyperparameters to ``parser``."""
        from mariha.replay.buffers import BufferType
        from mariha.utils.running import float_or_str, sci2int, str2bool

        # SAC core
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--polyak", type=float, default=0.995)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--lr_decay", type=str, default=None,
                            choices=[None, "exponential", "linear"])
        parser.add_argument("--lr_decay_rate", type=float, default=0.1)
        parser.add_argument("--alpha", type=float_or_str, default="auto",
                            help="Entropy coefficient ('auto' for automatic tuning).")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--replay_size", type=sci2int, default=int(1e5))
        parser.add_argument("--start_steps", type=sci2int, default=10_000)
        parser.add_argument("--update_after", type=sci2int, default=5_000)
        parser.add_argument("--update_every", type=int, default=50)
        parser.add_argument("--n_updates", type=int, default=50)
        parser.add_argument("--clipnorm", type=float, default=None)

        # Network architecture
        parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
        parser.add_argument("--activation", type=str, default="tanh",
                            choices=["tanh", "relu", "elu", "lrelu"])
        parser.add_argument("--use_layer_norm", type=str2bool, default=False)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--hide_task_id", type=str2bool, default=False)

        # Replay buffer
        parser.add_argument("--buffer_type", type=str, default="fifo",
                            choices=[bt.value for bt in BufferType])

        # Task-change behaviour
        parser.add_argument("--reset_buffer_on_task_change", type=str2bool, default=True)
        parser.add_argument("--reset_optimizer_on_task_change", type=str2bool, default=False)
        parser.add_argument("--reset_actor_on_task_change", type=str2bool, default=False)
        parser.add_argument("--reset_critic_on_task_change", type=str2bool, default=False)
        parser.add_argument("--agent_policy_exploration", type=str2bool, default=False)

        # Logging
        parser.add_argument("--log_every", type=sci2int, default=1000)
        parser.add_argument("--save_freq_epochs", type=int, default=25)
        parser.add_argument(
            "--render_every", type=int, default=0,
            help=(
                "If > 0, open a live window and play one full greedy episode every "
                "this many training episodes (a checkpoint render). 0 = disabled."
            ),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace, env, logger, scene_ids: list) -> "SAC":
        """Construct a SAC (or subclass) from parsed CLI args."""
        from mariha.rl import models
        from mariha.replay.buffers import BufferType
        from mariha.utils.running import get_activation_from_str, get_readable_timestamp

        activation = get_activation_from_str(getattr(args, "activation", "tanh"))
        policy_kwargs = dict(
            hidden_sizes=tuple(getattr(args, "hidden_sizes", [256, 256])),
            activation=activation,
            use_layer_norm=getattr(args, "use_layer_norm", False),
            num_heads=getattr(args, "num_heads", 1),
            hide_task_id=getattr(args, "hide_task_id", False),
        )

        experiment_dir = Path(getattr(args, "experiment_dir", "experiments"))
        timestamp = get_readable_timestamp()

        sac_kwargs = dict(
            env=env,
            logger=logger,
            scene_ids=scene_ids,
            agent=getattr(args, "agent", None),
            policy_kwargs=policy_kwargs,
            seed=getattr(args, "seed", 0),
            log_every=getattr(args, "log_every", 1000),
            replay_size=getattr(args, "replay_size", int(1e5)),
            gamma=getattr(args, "gamma", 0.99),
            polyak=getattr(args, "polyak", 0.995),
            lr=getattr(args, "lr", 1e-3),
            lr_decay=getattr(args, "lr_decay", None),
            lr_decay_rate=getattr(args, "lr_decay_rate", 0.1),
            alpha=getattr(args, "alpha", "auto"),
            batch_size=getattr(args, "batch_size", 128),
            start_steps=getattr(args, "start_steps", 10_000),
            update_after=getattr(args, "update_after", 5_000),
            update_every=getattr(args, "update_every", 50),
            n_updates=getattr(args, "n_updates", 50),
            save_freq_epochs=getattr(args, "save_freq_epochs", 25),
            reset_buffer_on_task_change=getattr(args, "reset_buffer_on_task_change", True),
            buffer_type=BufferType(getattr(args, "buffer_type", "fifo")),
            reset_optimizer_on_task_change=getattr(args, "reset_optimizer_on_task_change", False),
            reset_actor_on_task_change=getattr(args, "reset_actor_on_task_change", False),
            reset_critic_on_task_change=getattr(args, "reset_critic_on_task_change", False),
            clipnorm=getattr(args, "clipnorm", None),
            agent_policy_exploration=getattr(args, "agent_policy_exploration", False),
            render_every=getattr(args, "render_every", 0),
            post_burn_in_update_after=getattr(args, "post_burn_in_update_after", 0),
            buffer_mode=getattr(args, "buffer_mode", "single"),
            per_scene_capacity=getattr(args, "per_scene_capacity", 1000),
            flush_on=getattr(args, "flush_on", "session"),
            cl_hook_min_transitions=getattr(args, "cl_hook_min_transitions", 500),
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )

        return cls(**sac_kwargs)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full episode-driven training loop until the curriculum is exhausted."""
        self.start_time = time.time()

        # ---- initialise episode state ----
        try:
            obs, info = self.env.reset()
        except StopIteration:
            self.logger.log("Curriculum is empty — nothing to train.", color="red")
            return

        one_hot_vec: np.ndarray = info["task_one_hot"]
        current_scene_id: str = info.get("scene_id", "")
        current_session: str = info.get("session", "")
        current_task_idx: int = (
            self.scene_ids.index(current_scene_id)
            if current_scene_id in self.scene_ids
            else 0
        )

        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)
        self._handle_task_change(current_task_idx)

        episodes = 0
        episode_return = 0.0
        episode_len = 0
        global_timestep = 0
        task_timestep = 0  # resets on task change if agent_policy_exploration=False
        action_counts = {i: 0 for i in range(self.act_dim)}

        self.logger.log("Training started.", color="green")

        while True:
            # ---- action selection ----
            use_policy = (
                task_timestep >= self.start_steps
                or (self.agent_policy_exploration and current_task_idx > 0)
            )
            if use_policy:
                action = self.get_action_numpy(obs, one_hot_vec)
            else:
                action = self.env.action_space.sample()

            # ---- environment step ----
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_len += 1
            action_counts[action] += 1

            # For truncated episodes (timeout), don't store done=True — the
            # episode didn't actually end; the budget just ran out.
            done_to_store = terminated
            if self.buffer_mode == "per_scene":
                self.replay_buffer.store(
                    current_scene_id, obs, action, reward, next_obs,
                    done_to_store, one_hot_vec,
                )
            else:
                self.replay_buffer.store(
                    obs, action, reward, next_obs, done_to_store, one_hot_vec,
                )
            obs = next_obs

            # ---- end of episode ----
            if done:
                episodes += 1
                buf_pct = self.replay_buffer.size / self.replay_buffer.max_size * 100
                self.logger.log(
                    f"Ep {episodes:5d} | return={episode_return:7.2f} | "
                    f"len={episode_len:4d} | buf={buf_pct:.1f}%"
                )
                self.logger.store(
                    {
                        "train/return": episode_return,
                        "train/ep_length": episode_len,
                        "train/episodes": episodes,
                        "buffer_fill_pct": buf_pct,
                    }
                )

                episode_return = 0.0
                episode_len = 0

                if self.render_every > 0 and episodes % self.render_every == 0:
                    self.logger.log(
                        f"[render] episode {episodes} — opening live window...",
                        color="cyan",
                    )
                    self.env.render_checkpoint(self.get_action_numpy)

                if self.env.is_done:
                    break

                try:
                    obs, info = self.env.reset()
                except StopIteration:
                    break

                one_hot_vec = info["task_one_hot"]
                new_scene_id = info.get("scene_id", "")
                new_session = info.get("session", "")

                # Session boundary flush (per-scene buffer mode only).
                if (
                    self.buffer_mode == "per_scene"
                    and self.flush_on == "session"
                    and info.get("session_switch", False)
                ):
                    self._handle_session_boundary(current_task_idx)

                if info.get("task_switch", False):
                    # Gate CL hooks on minimum transitions in per-scene mode.
                    if self.buffer_mode == "per_scene":
                        buf_size = self.replay_buffer.total_size
                        if buf_size >= self.cl_hook_min_transitions:
                            self.on_task_end(current_task_idx)
                        else:
                            self.logger.log(
                                f"[CL] Skipping on_task_end — only {buf_size} "
                                f"transitions (need {self.cl_hook_min_transitions}).",
                                color="yellow",
                            )
                    else:
                        self.on_task_end(current_task_idx)
                    new_task_idx = (
                        self.scene_ids.index(new_scene_id)
                        if new_scene_id in self.scene_ids
                        else current_task_idx
                    )
                    self._handle_task_change(new_task_idx)
                    current_task_idx = new_task_idx
                    current_scene_id = new_scene_id
                    if not self.agent_policy_exploration:
                        task_timestep = 0

                current_session = new_session

            # ---- policy update ----
            if (
                global_timestep >= self.update_after
                and global_timestep % self.update_every == 0
                and self.replay_buffer.size >= self.batch_size
            ):
                t_update = time.time()
                for _ in range(self.n_updates):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    episodic_batch = self.get_episodic_batch(current_task_idx)
                    results = self.learn_on_batch(
                        tf.constant(current_task_idx), batch, episodic_batch
                    )
                    if self.buffer_type in (BufferType.PER, BufferType.PRIORITY):
                        self.replay_buffer.update_weights(
                            batch["idxs"].numpy(), results["abs_error"].numpy()
                        )
                    self._log_after_update(results)
                self.logger.log(
                    f"Updated in {time.time() - t_update:.2f}s (step {global_timestep})"
                )

            # ---- periodic logging and checkpointing ----
            if (global_timestep + 1) % self.log_every == 0:
                epoch = (global_timestep + 1) // self.log_every
                if epoch % self.save_freq_epochs == 0:
                    self.save_model(current_task_idx)
                self._log_after_epoch(epoch, global_timestep, action_counts)
                action_counts = {i: 0 for i in range(self.act_dim)}

            global_timestep += 1
            task_timestep += 1

        # ---- training complete ----
        self.on_task_end(current_task_idx)
        self.save_model(current_task_idx)
        self.logger.log(
            f"Training complete — {global_timestep} steps, {episodes} episodes.",
            color="green",
        )
