"""Proximal Policy Optimization for the MariHA continual-learning benchmark.

Architecture and hyperparameters are adapted from the ``ppo_study`` reference
implementation (PyTorch), translated to TensorFlow/Keras.

Key design choices:
- Shared CNN backbone (4x Conv2D(32, 3x3, stride=2, same) + Dense(512))
  with separate actor and critic heads.
- On-policy: no replay buffer; uses a fixed-length rollout buffer with GAE.
- Episode-driven loop: rollouts may span multiple episodes from the
  ``ContinualLearningEnv`` curriculum.
- Per-scene buffers (``buffer_mode``) are not applicable to PPO.

Usage::

    mariha-run-cl --agent ppo --subject sub-01 --seed 0 --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

from mariha.benchmark.agent import BenchmarkAgent
from mariha.replay.buffers import RolloutBuffer
from mariha.rl import models
from mariha.utils.logging import EpochLogger
from mariha.utils.running import set_seed


class PPO(BenchmarkAgent):
    """Proximal Policy Optimization for continual learning on MariHA.

    After construction, call ``run()`` to begin training.
    """

    def __init__(
        self,
        env,
        logger: EpochLogger,
        scene_ids: list,
        seed: int = 0,
        rollout_length: int = 512,
        n_epochs: int = 4,
        n_minibatches: int = 4,
        clip_ratio: float = 0.2,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        lr: float = 1e-4,
        vf_coef: float = 1.0,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        log_every: int = 1000,
        save_freq_epochs: int = 25,
        render_every: int = 0,
        reset_optimizer_on_task_change: bool = False,
        reset_network_on_task_change: bool = False,
        hidden_size: int = 512,
        hide_task_id: bool = False,
        experiment_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        set_seed(seed, env=env)

        self.env = env
        self.scene_ids = scene_ids
        self.num_tasks = len(scene_ids)
        self.logger = logger
        self.rollout_length = rollout_length
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.save_freq_epochs = save_freq_epochs
        self.render_every = render_every
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_network_on_task_change = reset_network_on_task_change
        self.experiment_dir = experiment_dir or Path("experiments")
        self.timestamp = timestamp

        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n
        logger.log(f"Observation shape: {self.obs_shape}", color="blue")
        logger.log(f"Action dim:        {self.act_dim}", color="blue")
        logger.log(f"Num tasks:         {self.num_tasks}", color="blue")

        # Network
        self.model = models.PPOActorCritic(
            state_space=env.observation_space,
            action_space=env.action_space,
            num_tasks=self.num_tasks,
            hidden_size=hidden_size,
            hide_task_id=hide_task_id,
        )

        self.optimizer = _keras.optimizers.legacy.Adam(learning_rate=lr)

        # Rollout buffer
        self.minibatch_size = rollout_length // n_minibatches
        self.rollout_buffer = RolloutBuffer(
            obs_shape=self.obs_shape,
            size=rollout_length,
            num_tasks=self.num_tasks,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.start_time: float = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(
        self, obs: np.ndarray, task_one_hot: np.ndarray, deterministic: bool = False
    ) -> int:
        obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
        one_hot_t = tf.expand_dims(tf.cast(task_one_hot, tf.float32), 0)
        logits, _ = self.model(obs_t, one_hot_t)
        if deterministic:
            return int(tf.argmax(logits[0]).numpy())
        probs = tf.nn.softmax(logits[0])
        return int(tf.random.categorical(tf.math.log(probs[tf.newaxis]), 1)[0, 0].numpy())

    def get_action_with_info(
        self, obs: np.ndarray, task_one_hot: np.ndarray
    ) -> tuple:
        """Return ``(action, log_prob, value)`` for rollout collection."""
        obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
        one_hot_t = tf.expand_dims(tf.cast(task_one_hot, tf.float32), 0)
        logits, value = self.model(obs_t, one_hot_t)
        probs = tf.nn.softmax(logits[0])
        dist = tf.random.categorical(tf.math.log(probs[tf.newaxis]), 1)
        action = int(dist[0, 0].numpy())
        log_prob = float(tf.math.log(probs[action] + 1e-8).numpy())
        return action, log_prob, float(value[0, 0].numpy())

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    @tf.function
    def _update_step(self, obs, actions, returns, advantages, old_log_probs, one_hot):
        """Single minibatch PPO gradient step."""
        with tf.GradientTape() as tape:
            logits, values = self.model(obs, one_hot)
            values = tf.squeeze(values, axis=-1)

            # New log probs
            probs = tf.nn.softmax(logits)
            log_probs_all = tf.math.log(probs + 1e-8)
            new_log_probs = tf.reduce_sum(
                log_probs_all * tf.one_hot(actions, self.act_dim), axis=-1
            )

            # PPO clipped objective
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )

            # Critic loss (Huber / smooth L1)
            critic_loss = tf.reduce_mean(
                tf.keras.losses.huber(returns, values, delta=1.0)
            )

            # Entropy bonus
            entropy = -tf.reduce_sum(probs * log_probs_all, axis=-1)
            entropy_loss = -tf.reduce_mean(entropy)

            total_loss = (
                actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
            )

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return dict(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            entropy=tf.reduce_mean(entropy),
            total_loss=total_loss,
        )

    def update(self) -> Dict:
        """Run ``n_epochs`` of minibatch updates on the current rollout."""
        metrics: Dict = {}
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_batches(self.minibatch_size):
                results = self._update_step(
                    batch["obs"],
                    batch["actions"],
                    batch["returns"],
                    batch["advantages"],
                    batch["old_log_probs"],
                    batch["one_hot"],
                )
                metrics = {k: float(v) for k, v in results.items()}
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full episode-driven PPO training loop."""
        self.start_time = time.time()

        try:
            obs, info = self.env.reset()
        except StopIteration:
            self.logger.log("Curriculum is empty — nothing to train.", color="red")
            return

        one_hot_vec = info["task_one_hot"]
        current_scene_id = info.get("scene_id", "")
        current_task_idx = (
            self.scene_ids.index(current_scene_id)
            if current_scene_id in self.scene_ids
            else 0
        )
        self.on_task_start(current_task_idx, current_scene_id)

        episodes = 0
        episode_return = 0.0
        episode_len = 0
        global_timestep = 0
        action_counts = {i: 0 for i in range(self.act_dim)}
        curriculum_done = False

        self.logger.log("PPO training started.", color="green")

        while not curriculum_done:
            # ---- collect rollout ----
            self.rollout_buffer.reset()
            for _ in range(self.rollout_length):
                action, log_prob, value = self.get_action_with_info(obs, one_hot_vec)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_len += 1
                action_counts[action] += 1

                self.rollout_buffer.store(
                    obs, action, reward, done, value, log_prob, one_hot_vec
                )
                obs = next_obs

                if done:
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

                    if self.render_every > 0 and episodes % self.render_every == 0:
                        self.env.render_checkpoint(self.get_action)

                    if self.env.is_done:
                        curriculum_done = True
                        break

                    try:
                        obs, info = self.env.reset()
                    except StopIteration:
                        curriculum_done = True
                        break

                    one_hot_vec = info["task_one_hot"]
                    new_scene_id = info.get("scene_id", "")

                    if info.get("task_switch", False):
                        self.on_task_end(current_task_idx)
                        new_task_idx = (
                            self.scene_ids.index(new_scene_id)
                            if new_scene_id in self.scene_ids
                            else current_task_idx
                        )
                        if self.reset_optimizer_on_task_change:
                            from mariha.utils.running import reset_optimizer
                            reset_optimizer(self.optimizer)
                        if self.reset_network_on_task_change:
                            from mariha.utils.running import reset_weights
                            reset_weights(
                                self.model, models.PPOActorCritic,
                                dict(
                                    state_space=self.env.observation_space,
                                    action_space=self.env.action_space,
                                    num_tasks=self.num_tasks,
                                ),
                            )
                        self.on_task_start(new_task_idx, new_scene_id)
                        current_task_idx = new_task_idx
                        current_scene_id = new_scene_id

                global_timestep += 1

            # ---- compute returns and advantages ----
            if self.rollout_buffer.ptr > 0:
                # Bootstrap value for the last observation.
                obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
                one_hot_t = tf.expand_dims(tf.cast(one_hot_vec, tf.float32), 0)
                _, last_value = self.model(obs_t, one_hot_t)
                last_val = float(last_value[0, 0].numpy())
                self.rollout_buffer.compute_returns_and_advantages(last_val)

                # ---- policy update ----
                t_update = time.time()
                metrics = self.update()
                self.logger.log(
                    f"PPO update | step {global_timestep} | "
                    f"loss={metrics.get('total_loss', 0):.4f} | "
                    f"{time.time() - t_update:.2f}s"
                )
                self.logger.store(
                    {
                        "train/loss_actor": metrics.get("actor_loss", 0),
                        "train/loss_critic": metrics.get("critic_loss", 0),
                        "train/entropy": metrics.get("entropy", 0),
                    }
                )

            # ---- periodic logging and checkpointing ----
            if (global_timestep + 1) % self.log_every == 0:
                epoch = (global_timestep + 1) // self.log_every
                if epoch % self.save_freq_epochs == 0:
                    self.save_model(current_task_idx)
                self._log_after_epoch(epoch, global_timestep, action_counts)
                action_counts = {i: 0 for i in range(self.act_dim)}

        # ---- training complete ----
        self.on_task_end(current_task_idx)
        self.save_model(current_task_idx)
        self.logger.log(
            f"PPO training complete — {global_timestep} steps, {episodes} episodes.",
            color="green",
        )

    # ------------------------------------------------------------------
    # Burn-in
    # ------------------------------------------------------------------

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        """Pre-train on a single scene using PPO rollouts.

        Collects rollouts on the burn-in scene and runs PPO updates.  After
        completion the rollout buffer is reset.

        Args:
            burn_in_spec: ``EpisodeSpec`` for the burn-in scene.
            num_steps: Total environment steps to collect.
        """
        from mariha.env.continual import make_scene_env
        from mariha.env.scenario_gen import load_metadata
        from mariha.env.base import SCENARIOS_DIR

        scene_id = burn_in_spec.scene_id
        self.logger.log(
            f"[burn-in] PPO burn-in on '{scene_id}' for {num_steps} steps.",
            color="cyan",
        )

        scene_meta = load_metadata(SCENARIOS_DIR)
        exit_point = scene_meta[scene_id]["exit_point"]
        burn_env = make_scene_env(
            scene_id=scene_id,
            exit_point=exit_point,
            scene_ids=self.scene_ids,
            render_mode=None,
        )

        step = 0
        episodes = 0
        t_start = time.time()
        try:
            obs, info = burn_env.reset(episode_spec=burn_in_spec)
            one_hot_vec = info["task_one_hot"]

            while step < num_steps:
                self.rollout_buffer.reset()
                steps_this_rollout = min(self.rollout_length, num_steps - step)
                for _ in range(steps_this_rollout):
                    action, log_prob, value = self.get_action_with_info(obs, one_hot_vec)
                    next_obs, reward, terminated, truncated, _info = burn_env.step(action)
                    done = terminated or truncated
                    self.rollout_buffer.store(
                        obs, action, reward, done, value, log_prob, one_hot_vec
                    )
                    obs = next_obs
                    step += 1

                    if done:
                        episodes += 1
                        obs, info = burn_env.reset(episode_spec=burn_in_spec)
                        one_hot_vec = info["task_one_hot"]

                if self.rollout_buffer.ptr > 0:
                    obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
                    one_hot_t = tf.expand_dims(tf.cast(one_hot_vec, tf.float32), 0)
                    _, last_value = self.model(obs_t, one_hot_t)
                    self.rollout_buffer.compute_returns_and_advantages(
                        float(last_value[0, 0].numpy())
                    )
                    self.update()
        finally:
            burn_env.close()
        self.rollout_buffer.reset()
        self.logger.log(
            f"[burn-in] Complete — {step} steps, {episodes} episodes "
            f"in {time.time() - t_start:.1f}s.",
            color="cyan",
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_after_epoch(
        self, epoch: int, global_timestep: int, action_counts: Dict
    ) -> None:
        self.logger.log_tabular("epoch", epoch)
        self.logger.log_tabular("total_env_steps", global_timestep + 1)
        self.logger.log_tabular("train/return", with_min_and_max=True)
        self.logger.log_tabular("train/ep_length", average_only=True)
        self.logger.log_tabular("train/episodes", average_only=True)
        self.logger.log_tabular("train/loss_actor", average_only=True)
        self.logger.log_tabular("train/loss_critic", average_only=True)
        self.logger.log_tabular("train/entropy", average_only=True)
        for i, count in action_counts.items():
            self.logger.log_tabular(f"train/actions/{i}", count)
        self.logger.log_tabular("walltime", time.time() - self.start_time)
        self.logger.dump_tabular()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_model(self, current_task_idx: int) -> None:
        ts = self.timestamp or ""
        model_dir = str(
            self.experiment_dir / "checkpoints" / "ppo" / f"{ts}_task{current_task_idx}"
        )
        self.logger.log(f"Saving PPO model to {model_dir}", color="crimson")
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_weights(os.path.join(model_dir, "ppo_actor_critic"))

    def save_checkpoint(self, directory: Path) -> None:
        os.makedirs(str(directory), exist_ok=True)
        self.model.save_weights(os.path.join(str(directory), "ppo_actor_critic"))

    def load_checkpoint(self, directory: Path) -> None:
        self.model.load_weights(os.path.join(str(directory), "ppo_actor_critic"))

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        from mariha.utils.running import str2bool

        parser.add_argument("--rollout_length", type=int, default=512)
        parser.add_argument("--n_epochs", type=int, default=4)
        parser.add_argument("--n_minibatches", type=int, default=4)
        parser.add_argument("--clip_ratio", type=float, default=0.2)
        parser.add_argument("--gae_lambda", type=float, default=0.95)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--vf_coef", type=float, default=1.0)
        parser.add_argument("--ent_coef", type=float, default=0.01)
        parser.add_argument("--max_grad_norm", type=float, default=0.5)
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--hide_task_id", type=str2bool, default=False)
        parser.add_argument(
            "--reset_optimizer_on_task_change", type=str2bool, default=False,
        )
        parser.add_argument(
            "--reset_network_on_task_change", type=str2bool, default=False,
        )
        parser.add_argument("--log_every", type=int, default=1000)
        parser.add_argument("--save_freq_epochs", type=int, default=25)
        parser.add_argument(
            "--render_every", type=int, default=0,
            help="Play a greedy episode every N training episodes (0=disabled).",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace, env, logger, scene_ids: list) -> "PPO":
        from mariha.utils.running import get_readable_timestamp

        experiment_dir = Path(getattr(args, "experiment_dir", "experiments"))
        return cls(
            env=env,
            logger=logger,
            scene_ids=scene_ids,
            seed=getattr(args, "seed", 0),
            rollout_length=getattr(args, "rollout_length", 512),
            n_epochs=getattr(args, "n_epochs", 4),
            n_minibatches=getattr(args, "n_minibatches", 4),
            clip_ratio=getattr(args, "clip_ratio", 0.2),
            gae_lambda=getattr(args, "gae_lambda", 0.95),
            gamma=getattr(args, "gamma", 0.99),
            lr=getattr(args, "lr", 1e-4),
            vf_coef=getattr(args, "vf_coef", 1.0),
            ent_coef=getattr(args, "ent_coef", 0.01),
            max_grad_norm=getattr(args, "max_grad_norm", 0.5),
            log_every=getattr(args, "log_every", 1000),
            save_freq_epochs=getattr(args, "save_freq_epochs", 25),
            render_every=getattr(args, "render_every", 0),
            reset_optimizer_on_task_change=getattr(args, "reset_optimizer_on_task_change", False),
            reset_network_on_task_change=getattr(args, "reset_network_on_task_change", False),
            hidden_size=getattr(args, "hidden_size", 512),
            hide_task_id=getattr(args, "hide_task_id", False),
            experiment_dir=experiment_dir,
            timestamp=get_readable_timestamp(),
        )
