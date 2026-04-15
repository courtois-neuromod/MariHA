"""Soft Actor-Critic for the MariHA continual-learning benchmark.

Discrete SAC with two critics, target networks, optional automatic
entropy tuning, and a flexible replay buffer (FIFO / reservoir / PER /
per-scene pool).  Built on :class:`mariha.rl.base.BaseAgent` with
``update_granularity = "per_step"``.

Continual-learning methods are attached via composition: assign an
instance of :class:`mariha.methods.base.CLMethod` to ``agent.cl_method``
after construction.  SAC invokes the CL method's hooks
(:meth:`compute_loss_penalty`, :meth:`adjust_gradients`,
:meth:`after_gradient_step`, :meth:`get_episodic_batch`) from inside its
per-task ``learn_on_batch`` ``tf.function``; the per-task recompile in
:meth:`on_task_change` keeps Python-time branches (e.g. "skip the
penalty on task 0") baked into the per-task graph.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf

# TF 2.17+ ships Keras 3 by default; tf_keras provides Keras 2 compatibility.
try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

from tensorflow_probability.python.distributions import Categorical

from mariha.replay.buffers import (
    BufferType,
    PrioritizedExperienceReplay,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReservoirReplayBuffer,
)
from mariha.rl import models
from mariha.rl.base import BaseAgent, run_burn_in
from mariha.utils.logging import EpochLogger
from mariha.utils.running import (
    create_one_hot_vec,
    reset_optimizer,
    reset_weights,
)


class SAC(BaseAgent):
    """Soft Actor-Critic for continual learning on MariHA.

    Per-step update style: stores every transition in the replay buffer
    and fires a batch of ``n_updates`` gradient steps every
    ``update_every`` environment steps once ``update_after`` steps have
    elapsed.

    Args:
        env: A ``ContinualLearningEnv`` instance.
        logger: An :class:`EpochLogger` for recording training statistics.
        run_ids: Ordered list of all BIDS run IDs (one per CL task).
        actor_cl: Actor class to instantiate.
        critic_cl: Critic class to instantiate.
        policy_kwargs: Keyword arguments forwarded to actor + critic ctors.
        seed: Global random seed.
        log_every: Number of env steps between logging epochs.
        replay_size: Replay buffer capacity.
        gamma: Discount factor.
        polyak: Polyak coefficient for target networks.
        lr: Learning rate.
        lr_decay: Schedule type: ``None``, ``'exponential'``, ``'linear'``.
        lr_decay_rate: Decay factor.
        lr_decay_steps: Steps over which to decay (heuristic if ``None``).
        alpha: Entropy coefficient (``'auto'`` for automatic tuning).
        batch_size: Mini-batch size for gradient updates.
        start_steps: Random-action warm-up before the policy is used.
        update_after: Min transitions in the buffer before updates start.
        update_every: Steps between consecutive update rounds.
        n_updates: Gradient updates per round.
        save_freq_epochs: Checkpointing frequency in logging epochs.
        reset_buffer_on_task_change: Clear the replay buffer at task switches.
        buffer_type: Which replay buffer implementation to use.
        reset_optimizer_on_task_change: Reset Adam slots on task change.
        reset_actor_on_task_change: Re-init actor on task change.
        reset_critic_on_task_change: Re-init critic on task change.
        clipnorm: Global gradient-norm clipping (``None`` = disabled).
        agent_policy_exploration: If ``True``, random exploration only on
            the first task; later tasks start with the trained policy.
        render_every: Render a greedy episode every N training episodes.
        post_burn_in_update_after: ``update_after`` value to apply after
            burn-in completes.
        buffer_mode: ``"single"`` (one global buffer) or ``"per_scene"``
            (a buffer pool with per-scene shards).
        per_scene_capacity: Capacity per scene in ``"per_scene"`` mode.
        flush_on: Trigger for per-scene buffer flush (``"session"``).
        cl_hook_min_transitions: Min transitions before CL ``on_task_end``
            fires in per-scene mode (kept for legacy parity).
        experiment_dir: Root directory for checkpoints and logs.
        timestamp: Run timestamp string.
    """

    update_granularity: str = "per_step"

    def __init__(
        self,
        env,
        logger: EpochLogger,
        run_ids: List[str],
        *,
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
        super().__init__(
            env=env,
            logger=logger,
            run_ids=run_ids,
            agent_name="sac",
            seed=seed,
            log_every=log_every,
            save_freq_epochs=save_freq_epochs,
            render_every=render_every,
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )

        if policy_kwargs is None:
            policy_kwargs = {}

        # ---- SAC hyperparameters ----
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
        self.reset_buffer_on_task_change = reset_buffer_on_task_change
        self.buffer_type = buffer_type
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_actor_on_task_change = reset_actor_on_task_change
        self.reset_critic_on_task_change = reset_critic_on_task_change
        self.clipnorm = clipnorm
        self.agent_policy_exploration = agent_policy_exploration
        self.post_burn_in_update_after = int(post_burn_in_update_after)
        self.buffer_mode = buffer_mode
        self.per_scene_capacity = per_scene_capacity
        self.flush_on = flush_on
        self.cl_hook_min_transitions = cl_hook_min_transitions

        # Share spaces + num_tasks with model constructors.
        policy_kwargs["state_space"] = env.observation_space
        policy_kwargs["action_space"] = env.action_space
        policy_kwargs["num_tasks"] = self.num_tasks

        # ---- replay buffer ----
        self._init_replay_buffer()

        # ---- networks ----
        self.actor_cl = actor_cl
        self.critic_cl = critic_cl
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

        # ---- optimizer with optional LR decay ----
        if lr_decay is not None and lr_decay_steps is None:
            lr_decay_steps = 500_000
        if lr_decay == "exponential":
            lr = _keras.optimizers.schedules.ExponentialDecay(
                lr, lr_decay_steps, lr_decay_rate
            )
        elif lr_decay == "linear":
            lr = _keras.optimizers.schedules.PolynomialDecay(
                lr, lr_decay_steps, lr * lr_decay_rate, power=1.0
            )
        self.optimizer = _keras.optimizers.legacy.Adam(learning_rate=lr)

        # ---- automatic entropy tuning ----
        self.auto_alpha = alpha == "auto"
        if self.auto_alpha:
            self.all_log_alpha = tf.Variable(
                np.zeros((self.num_tasks, 1), dtype=np.float32),
                trainable=True,
            )
            # Target entropy: fraction of maximum discrete entropy.
            self.target_entropy = 0.98 * np.log(self.act_dim)

        # ``learn_on_batch`` is recompiled per task in :meth:`on_task_change`
        # so the embedded ``current_task_idx`` constant matches.
        self.learn_on_batch: Optional[Callable] = None

    # ==================================================================
    # Replay buffer initialisation
    # ==================================================================

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
        if self.buffer_type == BufferType.FIFO:
            return ReplayBuffer
        if self.buffer_type == BufferType.RESERVOIR:
            return ReservoirReplayBuffer
        raise ValueError(
            f"Buffer type '{self.buffer_type.value}' is not supported with "
            "buffer_mode='per_scene'.  Use 'fifo' or 'reservoir'."
        )

    # ==================================================================
    # Agent-agnostic CL contract (Phase 4)
    # ==================================================================

    def get_named_parameter_groups(self) -> Dict[str, List[tf.Variable]]:
        """Named parameter groups for agent-agnostic CL methods.

        SAC exposes three disjoint groups whose union equals the
        optimizer's view of trainable parameters: ``actor``,
        ``critic1``, ``critic2``.  This is the alignment expected by
        :meth:`get_learn_on_batch` when it builds the
        ``grads_by_group`` dict it passes to
        ``cl_method.adjust_gradients``.
        """
        return {
            "actor": list(self.actor.trainable_variables),
            "critic1": list(self.critic1.trainable_variables),
            "critic2": list(self.critic2.trainable_variables),
        }

    def forward_for_importance(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Differentiable forward outputs per parameter group.

        Used by EWC/MAS-style CL methods to compute Jacobians of each
        group's output with respect to its parameters.  The keys mirror
        :meth:`get_named_parameter_groups`.
        """
        return {
            "actor": self.actor(obs, one_hot),
            "critic1": self.critic1(obs, one_hot),
            "critic2": self.critic2(obs, one_hot),
        }

    def distill_targets(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Distillation targets for DER/ClonEx-style CL methods.

        SAC exposes its actor logits *and* both critic Q-value vectors
        so a single :class:`DistillationMethod` instance can supply
        both DER (actor-only) and ClonEx (actor + critic) variants
        without an agent-specific code path.  The
        :class:`DistillationMethod` looks at which keys are present and
        which it has been configured to use.
        """
        return {
            "actor_logits": self.actor(obs, one_hot),
            "critic1_q": self.critic1(obs, one_hot),
            "critic2_q": self.critic2(obs, one_hot),
        }

    # ==================================================================
    # Action selection
    # ==================================================================

    def get_log_alpha(self, one_hot: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(
            tf.linalg.matmul(
                tf.expand_dims(tf.convert_to_tensor(one_hot), 1),
                self.all_log_alpha,
            )
        )

    @tf.function
    def _get_action_tf(
        self,
        obs: tf.Tensor,
        one_hot_task_id: tf.Tensor,
        deterministic: tf.Tensor = tf.constant(False),
    ) -> tf.Tensor:
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
        """BenchmarkAgent interface: numpy in, int out."""
        return self.get_action_numpy(obs, task_one_hot, deterministic)

    # ==================================================================
    # Gradient computation
    # ==================================================================

    def get_learn_on_batch(self, current_task_idx: int) -> Callable:
        """Return a compiled ``learn_on_batch`` function for ``current_task_idx``.

        ``current_task_idx`` is captured in the closure as a Python
        integer so the CL method's per-task branches (e.g.
        ``ParameterRegularizer.compute_loss_penalty`` returning zero
        on the first task) are evaluated at trace time and baked into
        the per-task graph.  ``learn_on_batch`` is recompiled at every
        task boundary in :meth:`on_task_change`.
        """

        @tf.function
        def learn_on_batch(
            batch: Dict[str, tf.Tensor],
            episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
        ) -> Dict:
            gradients, metrics = self.get_gradients(
                current_task_idx=current_task_idx, **batch
            )
            actor_grads, critic_grads, alpha_grad = gradients

            # Build the named-group view the CL contract expects.
            n1 = len(self.critic1.trainable_variables)
            grads_by_group: Dict[str, List[Optional[tf.Tensor]]] = {
                "actor": list(actor_grads),
                "critic1": list(critic_grads[:n1]),
                "critic2": list(critic_grads[n1:]),
            }
            if self.cl_method is not None:
                grads_by_group = self.cl_method.adjust_gradients(
                    self,
                    grads_by_group,
                    task_idx=current_task_idx,
                    episodic_batch=episodic_batch,
                )

            # Reassemble for the optimizer.
            actor_grads = grads_by_group["actor"]
            critic_grads = list(grads_by_group["critic1"]) + list(
                grads_by_group["critic2"]
            )

            if self.clipnorm is not None:
                actor_grads = list(
                    tf.clip_by_global_norm(actor_grads, self.clipnorm)[0]
                )
                critic_grads = list(
                    tf.clip_by_global_norm(critic_grads, self.clipnorm)[0]
                )
                if alpha_grad is not None:
                    alpha_grad = tf.clip_by_norm(alpha_grad, self.clipnorm)
                # Refresh the group view so ``after_gradient_step`` sees
                # the gradients that were actually applied.
                grads_by_group = {
                    "actor": actor_grads,
                    "critic1": critic_grads[:n1],
                    "critic2": critic_grads[n1:],
                }

            self.apply_update(actor_grads, critic_grads, alpha_grad)

            if self.cl_method is not None:
                self.cl_method.after_gradient_step(
                    self,
                    grads_by_group,
                    task_idx=current_task_idx,
                    metrics=metrics,
                )

            return metrics

        return learn_on_batch

    def get_gradients(
        self,
        *,
        current_task_idx: int,
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
        one_hot: tf.Tensor,
        **kwargs,
    ) -> Tuple[Tuple, Dict]:
        """Compute SAC actor, critic, and alpha gradients.

        ``current_task_idx`` is a Python int captured by
        ``learn_on_batch``'s closure (one trace per task).  It is
        forwarded to ``cl_method.compute_loss_penalty`` so that
        regularizer-style CL methods can return zero on the first
        task at trace time.
        """
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

            min_q = dist.probs_parameter() * tf.stop_gradient(
                tf.minimum(q1, q2)
            )
            min_target_q = dist_next.probs_parameter() * tf.minimum(
                target_q1, target_q2
            )

            q_backup = tf.stop_gradient(
                rewards
                + self.gamma
                * (1 - done)
                * (
                    tf.math.reduce_sum(min_target_q, axis=-1)
                    - log_alpha_exp * entropy_next
                )
            )

            abs_error = tf.stop_gradient(
                tf.math.minimum(
                    tf.abs(q_backup - q1_vals), tf.abs(q_backup - q2_vals)
                )
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

            if self.cl_method is not None:
                auxiliary_loss = self.cl_method.compute_loss_penalty(
                    self, task_idx=current_task_idx
                )
            else:
                auxiliary_loss = tf.zeros([])
            # Apply the penalty to both losses; SAC's actor and critic
            # parameter groups are disjoint so there is no double-counting
            # of the penalty's gradient w.r.t. any parameter.
            actor_loss += auxiliary_loss
            value_loss += auxiliary_loss

        actor_gradients = g.gradient(
            actor_loss, self.actor.trainable_variables
        )
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        alpha_gradient = (
            g.gradient(alpha_loss, self.all_log_alpha)
            if self.auto_alpha
            else None
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

    def _compute_reference_loss(
        self, batch: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Per-group SAC losses on ``batch`` with no CL penalty.

        Used by gradient-projection CL methods (AGEM) to compute
        reference gradients on episodic memory.  Mirrors the loss
        computation inside :meth:`get_gradients` but skips the
        ``cl_method.compute_loss_penalty`` term and the alpha-loss
        side-effect — AGEM only projects actor and critic gradients.
        """
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done = batch["done"]
        one_hot = batch["one_hot"]

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
        min_q = dist.probs_parameter() * tf.stop_gradient(
            tf.minimum(q1, q2)
        )
        min_target_q = dist_next.probs_parameter() * tf.minimum(
            target_q1, target_q2
        )
        q_backup = tf.stop_gradient(
            rewards
            + self.gamma
            * (1 - done)
            * (
                tf.math.reduce_sum(min_target_q, axis=-1)
                - log_alpha_exp * entropy_next
            )
        )
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_vals) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_vals) ** 2)
        actor_loss = -tf.reduce_mean(
            log_alpha_exp * entropy + tf.reduce_sum(min_q, axis=-1)
        )
        return {
            "actor": actor_loss,
            "critic1": q1_loss,
            "critic2": q2_loss,
        }

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
            self.optimizer.apply_gradients(
                [(alpha_gradient, self.all_log_alpha)]
            )

        for v, tv in zip(
            self.critic1.trainable_variables,
            self.target_critic1.trainable_variables,
        ):
            tv.assign(self.polyak * tv + (1 - self.polyak) * v)
        for v, tv in zip(
            self.critic2.trainable_variables,
            self.target_critic2.trainable_variables,
        ):
            tv.assign(self.polyak * tv + (1 - self.polyak) * v)

    # ==================================================================
    # BaseAgent callback surface
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
        """Random warm-up for ``start_steps`` then policy.

        ``agent_policy_exploration=True`` confines the random warm-up to
        the very first task; subsequent tasks immediately use the trained
        policy.
        """
        use_policy = task_step >= self.start_steps or (
            self.agent_policy_exploration and current_task_idx > 0
        )
        if use_policy:
            action = self.get_action_numpy(obs, one_hot)
        else:
            # Sample directly from act_dim — works during burn-in when
            # the main env has been released (stable-retro single-emulator
            # constraint).
            action = int(np.random.randint(0, self.act_dim))
        return int(action), {}

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
        # For truncated episodes (timeout), don't store done=True — the
        # episode didn't actually end; the budget just ran out.
        done_to_store = terminated
        if self.buffer_mode == "per_scene":
            self.replay_buffer.store(
                scene_id, obs, action, reward, next_obs,
                done_to_store, one_hot,
            )
        else:
            self.replay_buffer.store(
                obs, action, reward, next_obs, done_to_store, one_hot,
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
        if self.learn_on_batch is None:
            # Lazy compile in case ``on_task_change`` hasn't fired yet
            # (e.g. update_after=0 first-task case).
            self.learn_on_batch = self.get_learn_on_batch(current_task_idx)

        t_update = time.time()
        for _ in range(self.n_updates):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            episodic_batch = (
                self.cl_method.get_episodic_batch(
                    self, task_idx=current_task_idx
                )
                if self.cl_method is not None
                else None
            )
            results = self.learn_on_batch(batch, episodic_batch)
            if self.buffer_type in (BufferType.PER, BufferType.PRIORITY):
                self.replay_buffer.update_weights(
                    batch["idxs"].numpy(), results["abs_error"].numpy()
                )
            self._log_after_update(results, current_task_idx)
        self.logger.log(
            f"Updated in {time.time() - t_update:.2f}s (step {global_step})"
        )

    # ==================================================================
    # Logging hooks
    # ==================================================================

    def _log_after_update(self, results: Dict, current_task_idx: int) -> None:
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
            self.logger.store(
                {
                    "train/alpha": float(
                        tf.math.exp(self.all_log_alpha[current_task_idx][0])
                    )
                }
            )
            self.logger.store(
                {
                    f"train/alpha/{task_idx}": float(
                        tf.math.exp(self.all_log_alpha[task_idx][0])
                    )
                    for task_idx in range(self.num_tasks)
                },
                display=False,
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
        try:
            buf_pct = (
                self.replay_buffer.size / self.replay_buffer.max_size * 100
            )
        except (AttributeError, ZeroDivisionError):
            buf_pct = 0.0
        self.logger.store({"buffer_fill_pct": buf_pct})

    def get_log_tabular_keys(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("buffer_fill_pct", {"average_only": True}),
            ("train/loss_pi", {"average_only": True}),
            ("train/loss_q1", {"average_only": True}),
            ("train/loss_q2", {"average_only": True}),
            ("train/loss_kl", {"average_only": True}),
            ("train/loss_reg", {"average_only": True}),
        ]

    # ==================================================================
    # Task / session lifecycle
    # ==================================================================

    def on_task_start(
        self, current_task_idx: int, run_id: str = ""
    ) -> None:
        """SAC's task-start log line + forward to attached CL method."""
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
        """SAC's task-end log line + forward to attached CL method."""
        self.logger.log(f"Task end:   idx={current_task_idx}", color="white")
        super().on_task_end(current_task_idx)

    def on_task_change(
        self, new_task_idx: int, new_run_id: str
    ) -> None:
        """Reset buffers/weights/optimizer + recompile ``learn_on_batch``.

        Called by the runner *after* :meth:`on_task_start`, which has
        already forwarded the task-start signal to the attached CL
        method.  This hook handles the SAC-internal state reset and the
        per-task ``tf.function`` recompile.
        """
        if self.reset_buffer_on_task_change and self.buffer_mode != "per_scene":
            self._init_replay_buffer()

        if self.reset_actor_on_task_change:
            reset_weights(self.actor, self.actor_cl, self.actor_kwargs)

        if self.reset_critic_on_task_change:
            reset_weights(
                self.critic1, self.critic_cl, self.policy_kwargs
            )
            self.target_critic1.set_weights(self.critic1.get_weights())
            reset_weights(
                self.critic2, self.critic_cl, self.policy_kwargs
            )
            self.target_critic2.set_weights(self.critic2.get_weights())

        if self.reset_optimizer_on_task_change:
            self.logger.log("Resetting optimizer.", color="cyan")
            reset_optimizer(self.optimizer)

        # Recompile so the embedded ``current_task_idx`` constant matches.
        self.learn_on_batch = self.get_learn_on_batch(new_task_idx)

    def handle_session_boundary(self, current_task_idx: int) -> None:
        """Flush per-scene buffers and run end-of-session gradient updates."""
        if self.buffer_mode != "per_scene" or self.flush_on != "session":
            return

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
                episodic_batch = (
                    self.cl_method.get_episodic_batch(
                        self, task_idx=current_task_idx
                    )
                    if self.cl_method is not None
                    else None
                )
                results = self.learn_on_batch(batch, episodic_batch)
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

        flushed = pool.flush_all()
        self.logger.log(
            f"[session-flush] Flushed {flushed} transitions.", color="cyan"
        )

    # ==================================================================
    # Checkpointing (BaseAgent contract)
    # ==================================================================

    def save_weights(self, directory: Path) -> None:
        directory = Path(directory)
        self.actor.save_weights(str(directory / "actor"))
        self.critic1.save_weights(str(directory / "critic1"))
        self.target_critic1.save_weights(str(directory / "target_critic1"))
        self.critic2.save_weights(str(directory / "critic2"))
        self.target_critic2.save_weights(str(directory / "target_critic2"))

    def load_weights(self, directory: Path) -> None:
        directory = Path(directory)
        self.actor.load_weights(str(directory / "actor"))
        self.critic1.load_weights(str(directory / "critic1"))
        self.target_critic1.load_weights(str(directory / "target_critic1"))
        self.critic2.load_weights(str(directory / "critic2"))
        self.target_critic2.load_weights(str(directory / "target_critic2"))

    # ==================================================================
    # Burn-in
    # ==================================================================

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        run_burn_in(self, burn_in_spec, num_steps)

    def on_burn_in_start(self, task_idx: int) -> None:
        """Compile ``learn_on_batch`` for the burn-in scene."""
        self.learn_on_batch = self.get_learn_on_batch(task_idx)

    def on_burn_in_end(self) -> None:
        """Reset the buffer and promote ``post_burn_in_update_after``."""
        self._init_replay_buffer()
        self.update_after = self.post_burn_in_update_after
        self.logger.log(
            f"[burn-in] update_after set to {self.update_after} for curriculum.",
            color="cyan",
        )

    # ==================================================================
    # Config interface
    # ==================================================================

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add SAC and CL-method hyperparameters to ``parser``."""
        from mariha.replay.buffers import BufferType
        from mariha.utils.running import float_or_str, sci2int, str2bool

        # SAC core
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--polyak", type=float, default=0.995)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--lr_decay",
            type=str,
            default=None,
            choices=[None, "exponential", "linear"],
        )
        parser.add_argument("--lr_decay_rate", type=float, default=0.1)
        parser.add_argument(
            "--alpha",
            type=float_or_str,
            default="auto",
            help="Entropy coefficient ('auto' for automatic tuning).",
        )
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--replay_size", type=sci2int, default=int(1e5))
        parser.add_argument("--start_steps", type=sci2int, default=10_000)
        parser.add_argument("--update_after", type=sci2int, default=5_000)
        parser.add_argument("--update_every", type=int, default=50)
        parser.add_argument("--n_updates", type=int, default=50)
        parser.add_argument("--clipnorm", type=float, default=None)

        # Network architecture
        parser.add_argument(
            "--hidden_sizes", type=int, nargs="+", default=[256, 256]
        )
        parser.add_argument(
            "--activation",
            type=str,
            default="tanh",
            choices=["tanh", "relu", "elu", "lrelu"],
        )
        parser.add_argument("--use_layer_norm", type=str2bool, default=False)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--hide_task_id", type=str2bool, default=False)

        # Replay buffer
        parser.add_argument(
            "--buffer_type",
            type=str,
            default="fifo",
            choices=[bt.value for bt in BufferType],
        )

        # Task-change behaviour
        parser.add_argument(
            "--reset_buffer_on_task_change", type=str2bool, default=True
        )
        parser.add_argument(
            "--reset_optimizer_on_task_change", type=str2bool, default=False
        )
        parser.add_argument(
            "--reset_actor_on_task_change", type=str2bool, default=False
        )
        parser.add_argument(
            "--reset_critic_on_task_change", type=str2bool, default=False
        )
        parser.add_argument(
            "--agent_policy_exploration", type=str2bool, default=False
        )

        # Logging
        parser.add_argument("--log_every", type=sci2int, default=1000)
        parser.add_argument("--save_freq_epochs", type=int, default=25)
        parser.add_argument(
            "--render_every",
            type=int,
            default=0,
            help=(
                "If > 0, open a live window and play one full greedy episode "
                "every this many training episodes (a checkpoint render). "
                "0 = disabled."
            ),
        )

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, env, logger, run_ids: list
    ) -> "SAC":
        """Construct a SAC (or subclass) from parsed CLI args."""
        from mariha.replay.buffers import BufferType
        from mariha.utils.running import (
            get_activation_from_str,
            get_readable_timestamp,
        )

        activation = get_activation_from_str(getattr(args, "activation", "tanh"))
        policy_kwargs = dict(
            hidden_sizes=tuple(getattr(args, "hidden_sizes", [256, 256])),
            activation=activation,
            use_layer_norm=getattr(args, "use_layer_norm", False),
            num_heads=getattr(args, "num_heads", 1),
            hide_task_id=getattr(args, "hide_task_id", False),
        )

        experiment_dir = Path(getattr(args, "experiment_dir", "experiments"))
        # Prefer the timestamp the benchmark context already composed (so the
        # checkpoint dir leaf shares its prefix with the run dir leaf and
        # ``mariha-evaluate --run_prefix`` can find both).  Fall back to a
        # fresh seeded timestamp for ad-hoc constructions (tests, scripts).
        timestamp = getattr(args, "run_timestamp", None) or (
            f"{get_readable_timestamp()}_seed{getattr(args, 'seed', 0)}"
        )

        sac_kwargs = dict(
            env=env,
            logger=logger,
            run_ids=run_ids,
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
            reset_buffer_on_task_change=getattr(
                args, "reset_buffer_on_task_change", True
            ),
            buffer_type=BufferType(getattr(args, "buffer_type", "fifo")),
            reset_optimizer_on_task_change=getattr(
                args, "reset_optimizer_on_task_change", False
            ),
            reset_actor_on_task_change=getattr(
                args, "reset_actor_on_task_change", False
            ),
            reset_critic_on_task_change=getattr(
                args, "reset_critic_on_task_change", False
            ),
            clipnorm=getattr(args, "clipnorm", None),
            agent_policy_exploration=getattr(
                args, "agent_policy_exploration", False
            ),
            render_every=getattr(args, "render_every", 0),
            post_burn_in_update_after=getattr(
                args, "post_burn_in_update_after", 0
            ),
            buffer_mode=getattr(args, "buffer_mode", "single"),
            per_scene_capacity=getattr(args, "per_scene_capacity", 1000),
            flush_on=getattr(args, "flush_on", "session"),
            cl_hook_min_transitions=getattr(
                args, "cl_hook_min_transitions", 500
            ),
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )

        return cls(**sac_kwargs)
