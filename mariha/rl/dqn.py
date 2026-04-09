"""Deep Q-Network (with Double DQN option) for the MariHA benchmark.

Structurally similar to SAC: off-policy, replay buffer, per-step updates.
Uses epsilon-greedy exploration instead of entropy-based.  Supports the
per-scene buffer mode from TODO 3.

Usage::

    mariha-run-cl --algorithm dqn  --subject sub-01 --seed 0 --lr 1e-4
    mariha-run-cl --algorithm ddqn --subject sub-01 --seed 0 --lr 1e-4
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
from mariha.replay.buffers import BufferType, ReplayBuffer, ReservoirReplayBuffer
from mariha.rl import models
from mariha.utils.logging import EpochLogger
from mariha.utils.running import set_seed


class DQN(BenchmarkAgent):
    """Deep Q-Network for continual learning on MariHA.

    Uses the same CNN architecture as SAC (``MlpCritic``) for the Q-network
    and target network with Polyak averaging.

    When ``double_dqn=True`` (the default), action selection comes from the
    online network while Q-value evaluation uses the target network.
    """

    def __init__(
        self,
        env,
        logger: EpochLogger,
        scene_ids: list,
        seed: int = 0,
        # DQN-specific
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50_000,
        double_dqn: bool = True,
        # Shared with SAC
        replay_size: int = 100_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-4,
        batch_size: int = 32,
        update_after: int = 5_000,
        update_every: int = 4,
        n_updates: int = 1,
        # Buffer params
        buffer_mode: str = "single",
        buffer_type: BufferType = BufferType.FIFO,
        per_scene_capacity: int = 1000,
        flush_on: str = "session",
        cl_hook_min_transitions: int = 500,
        reset_buffer_on_task_change: bool = True,
        # Task-change behaviour
        reset_optimizer_on_task_change: bool = False,
        reset_network_on_task_change: bool = False,
        # Architecture
        hidden_sizes: tuple = (256, 256),
        hide_task_id: bool = False,
        # Logging
        log_every: int = 1000,
        save_freq_epochs: int = 25,
        render_every: int = 0,
        post_burn_in_update_after: int = 0,
        experiment_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        set_seed(seed, env=env)

        self.env = env
        self.scene_ids = scene_ids
        self.num_tasks = len(scene_ids)
        self.logger = logger
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.double_dqn = double_dqn
        self.replay_size = int(replay_size)
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.n_updates = int(n_updates)
        self.buffer_mode = buffer_mode
        self.buffer_type = buffer_type
        self.per_scene_capacity = per_scene_capacity
        self.flush_on = flush_on
        self.cl_hook_min_transitions = cl_hook_min_transitions
        self.reset_buffer_on_task_change = reset_buffer_on_task_change
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_network_on_task_change = reset_network_on_task_change
        self.hidden_sizes = hidden_sizes
        self.hide_task_id = hide_task_id
        self.log_every = int(log_every)
        self.save_freq_epochs = save_freq_epochs
        self.render_every = int(render_every)
        self.post_burn_in_update_after = int(post_burn_in_update_after)
        self.experiment_dir = experiment_dir or Path("experiments")
        self.timestamp = timestamp

        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n
        logger.log(f"Observation shape: {self.obs_shape}", color="blue")
        logger.log(f"Action dim:        {self.act_dim}", color="blue")

        policy_kwargs = dict(
            state_space=env.observation_space,
            action_space=env.action_space,
            num_tasks=self.num_tasks,
            hidden_sizes=hidden_sizes,
            hide_task_id=hide_task_id,
        )

        # Replay buffer
        self._init_replay_buffer()

        # Networks: online Q and target Q (reuse MlpCritic from SAC)
        self.q_network = models.MlpCritic(**policy_kwargs)
        self.target_q_network = models.MlpCritic(**policy_kwargs)
        self.target_q_network.set_weights(self.q_network.get_weights())

        self.optimizer = _keras.optimizers.legacy.Adam(learning_rate=lr)

        self.start_time: float = 0.0

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def _init_replay_buffer(self) -> None:
        if self.buffer_mode == "per_scene":
            from mariha.replay.buffers import PerSceneBufferPool
            buf_cls = ReservoirReplayBuffer if self.buffer_type == BufferType.RESERVOIR else ReplayBuffer
            self.replay_buffer = PerSceneBufferPool(
                obs_shape=self.obs_shape,
                per_scene_capacity=self.per_scene_capacity,
                num_tasks=self.num_tasks,
                buffer_factory=lambda shape, cap, nt: buf_cls(shape, cap, nt),
            )
        else:
            if self.buffer_type == BufferType.RESERVOIR:
                self.replay_buffer = ReservoirReplayBuffer(
                    self.obs_shape, self.replay_size, self.num_tasks,
                )
            else:
                self.replay_buffer = ReplayBuffer(
                    self.obs_shape, self.replay_size, self.num_tasks,
                )

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def _get_epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(
        self, obs: np.ndarray, task_one_hot: np.ndarray, deterministic: bool = False
    ) -> int:
        obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
        one_hot_t = tf.expand_dims(tf.cast(task_one_hot, tf.float32), 0)
        q_values = self.q_network(obs_t, one_hot_t)
        return int(tf.argmax(q_values[0]).numpy())

    def _get_action_eps_greedy(
        self, obs: np.ndarray, one_hot: np.ndarray, epsilon: float
    ) -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        return self.get_action(obs, one_hot, deterministic=True)

    # ------------------------------------------------------------------
    # DQN update
    # ------------------------------------------------------------------

    @tf.function
    def _learn_on_batch(self, obs, actions, rewards, next_obs, done, one_hot):
        """Single DQN gradient step."""
        with tf.GradientTape() as tape:
            # Current Q-values for taken actions
            q_values = self.q_network(obs, one_hot)
            q_taken = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.act_dim), axis=-1
            )

            # Target Q-values
            if self.double_dqn:
                # DDQN: select action from online net, evaluate with target
                next_q_online = self.q_network(next_obs, one_hot)
                next_actions = tf.argmax(next_q_online, axis=-1, output_type=tf.int32)
                next_q_target = self.target_q_network(next_obs, one_hot)
                next_q = tf.reduce_sum(
                    next_q_target * tf.one_hot(next_actions, self.act_dim), axis=-1
                )
            else:
                next_q_target = self.target_q_network(next_obs, one_hot)
                next_q = tf.reduce_max(next_q_target, axis=-1)

            target = rewards + self.gamma * next_q * (1.0 - done)
            target = tf.stop_gradient(target)

            loss = tf.reduce_mean(tf.square(q_taken - target))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables)
        )

        # Polyak averaging for target network
        for v, tv in zip(
            self.q_network.trainable_variables,
            self.target_q_network.trainable_variables,
        ):
            tv.assign(self.polyak * tv + (1.0 - self.polyak) * v)

        return dict(loss=loss, q_mean=tf.reduce_mean(q_taken))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full episode-driven DQN training loop."""
        self.start_time = time.time()

        try:
            obs, info = self.env.reset()
        except StopIteration:
            self.logger.log("Curriculum is empty — nothing to train.", color="red")
            return

        one_hot_vec = info["task_one_hot"]
        current_scene_id = info.get("scene_id", "")
        current_session = info.get("session", "")
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

        self.logger.log("DQN training started.", color="green")

        while True:
            # ---- action selection ----
            epsilon = self._get_epsilon(global_timestep)
            action = self._get_action_eps_greedy(obs, one_hot_vec, epsilon)

            # ---- environment step ----
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_len += 1
            action_counts[action] += 1

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
                buf_pct = self.replay_buffer.size / max(self.replay_buffer.max_size, 1) * 100
                self.logger.log(
                    f"Ep {episodes:5d} | return={episode_return:7.2f} | "
                    f"len={episode_len:4d} | eps={epsilon:.3f} | buf={buf_pct:.1f}%"
                )
                self.logger.store(
                    {
                        "train/return": episode_return,
                        "train/ep_length": episode_len,
                        "train/episodes": episodes,
                        "train/epsilon": epsilon,
                        "buffer_fill_pct": buf_pct,
                    }
                )
                episode_return = 0.0
                episode_len = 0

                if self.render_every > 0 and episodes % self.render_every == 0:
                    self.env.render_checkpoint(self.get_action)

                if self.env.is_done:
                    break

                try:
                    obs, info = self.env.reset()
                except StopIteration:
                    break

                one_hot_vec = info["task_one_hot"]
                new_scene_id = info.get("scene_id", "")
                new_session = info.get("session", "")

                # Session boundary flush
                if (
                    self.buffer_mode == "per_scene"
                    and self.flush_on == "session"
                    and info.get("session_switch", False)
                ):
                    self._handle_session_boundary(current_task_idx)

                if info.get("task_switch", False):
                    self.on_task_end(current_task_idx)
                    new_task_idx = (
                        self.scene_ids.index(new_scene_id)
                        if new_scene_id in self.scene_ids
                        else current_task_idx
                    )
                    self._handle_task_change(new_task_idx, new_scene_id)
                    current_task_idx = new_task_idx
                    current_scene_id = new_scene_id

                current_session = new_session

            # ---- policy update ----
            if (
                global_timestep >= self.update_after
                and global_timestep % self.update_every == 0
                and self.replay_buffer.size >= self.batch_size
            ):
                for _ in range(self.n_updates):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    results = self._learn_on_batch(
                        batch["obs"],
                        batch["actions"],
                        batch["rewards"],
                        batch["next_obs"],
                        batch["done"],
                        batch["one_hot"],
                    )
                    self.logger.store(
                        {
                            "train/loss_q": float(results["loss"]),
                            "train/q_mean": float(results["q_mean"]),
                        }
                    )

            # ---- periodic logging and checkpointing ----
            if (global_timestep + 1) % self.log_every == 0:
                epoch = (global_timestep + 1) // self.log_every
                if epoch % self.save_freq_epochs == 0:
                    self.save_model(current_task_idx)
                self._log_after_epoch(epoch, global_timestep, action_counts)
                action_counts = {i: 0 for i in range(self.act_dim)}

            global_timestep += 1

        # ---- training complete ----
        self.on_task_end(current_task_idx)
        self.save_model(current_task_idx)
        self.logger.log(
            f"DQN training complete — {global_timestep} steps, {episodes} episodes.",
            color="green",
        )

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _handle_task_change(self, new_task_idx: int, new_scene_id: str) -> None:
        if self.reset_buffer_on_task_change and self.buffer_mode != "per_scene":
            self._init_replay_buffer()
        if self.reset_optimizer_on_task_change:
            from mariha.utils.running import reset_optimizer
            reset_optimizer(self.optimizer)
        if self.reset_network_on_task_change:
            from mariha.utils.running import reset_weights
            policy_kwargs = dict(
                state_space=self.env.observation_space,
                action_space=self.env.action_space,
                num_tasks=self.num_tasks,
                hidden_sizes=self.hidden_sizes,
                hide_task_id=self.hide_task_id,
            )
            reset_weights(self.q_network, models.MlpCritic, policy_kwargs)
            self.target_q_network.set_weights(self.q_network.get_weights())
        self.on_task_start(new_task_idx, new_scene_id)

    def _handle_session_boundary(self, current_task_idx: int) -> None:
        """Flush per-scene buffers with gradient updates at session boundary."""
        from mariha.replay.buffers import PerSceneBufferPool

        pool: PerSceneBufferPool = self.replay_buffer  # type: ignore[assignment]
        total = pool.total_size
        if total >= self.batch_size:
            n_updates = total // self.batch_size
            self.logger.log(
                f"[session-flush] DQN: {total} transitions — {n_updates} updates.",
                color="cyan",
            )
            for _ in range(n_updates):
                batch = pool.sample_batch(self.batch_size)
                self._learn_on_batch(
                    batch["obs"], batch["actions"], batch["rewards"],
                    batch["next_obs"], batch["done"], batch["one_hot"],
                )
        self.on_task_end(current_task_idx)
        flushed = pool.flush_all()
        self.logger.log(
            f"[session-flush] Flushed {flushed} transitions.", color="cyan"
        )

    # ------------------------------------------------------------------
    # Burn-in
    # ------------------------------------------------------------------

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        """Pre-train on a single scene with epsilon-greedy exploration.

        After completion the buffer is flushed and ``update_after`` is set
        to ``post_burn_in_update_after``.
        """
        from mariha.env.continual import make_scene_env
        from mariha.env.scenario_gen import load_metadata
        from mariha.env.base import SCENARIOS_DIR

        scene_id = burn_in_spec.scene_id
        self.logger.log(
            f"[burn-in] DQN burn-in on '{scene_id}' for {num_steps} steps.",
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

        obs, info = burn_env.reset(episode_spec=burn_in_spec)
        one_hot_vec = info["task_one_hot"]
        step = 0
        episodes = 0
        t_start = time.time()

        while step < num_steps:
            epsilon = self._get_epsilon(step)
            action = self._get_action_eps_greedy(obs, one_hot_vec, epsilon)
            next_obs, reward, terminated, truncated, _info = burn_env.step(action)
            done = terminated or truncated

            self.replay_buffer.store(
                obs, action, reward, next_obs, terminated, one_hot_vec
            )
            obs = next_obs

            if done:
                episodes += 1
                obs, info = burn_env.reset(episode_spec=burn_in_spec)
                one_hot_vec = info["task_one_hot"]

            if (
                step >= self.update_after
                and step % self.update_every == 0
                and self.replay_buffer.size >= self.batch_size
            ):
                for _ in range(self.n_updates):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self._learn_on_batch(
                        batch["obs"], batch["actions"], batch["rewards"],
                        batch["next_obs"], batch["done"], batch["one_hot"],
                    )

            step += 1

        burn_env.close()
        self._init_replay_buffer()
        self.update_after = self.post_burn_in_update_after
        self.logger.log(
            f"[burn-in] Complete — {step} steps, {episodes} episodes "
            f"in {time.time() - t_start:.1f}s. update_after={self.update_after}",
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
        self.logger.log_tabular("train/epsilon", average_only=True)
        self.logger.log_tabular("buffer_fill_pct", average_only=True)
        self.logger.log_tabular("train/loss_q", average_only=True)
        self.logger.log_tabular("train/q_mean", average_only=True)
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
            self.experiment_dir / "checkpoints" / "dqn" / f"{ts}_task{current_task_idx}"
        )
        self.logger.log(f"Saving DQN model to {model_dir}", color="crimson")
        os.makedirs(model_dir, exist_ok=True)
        self.q_network.save_weights(os.path.join(model_dir, "q_network"))
        self.target_q_network.save_weights(os.path.join(model_dir, "target_q_network"))

    def save_checkpoint(self, directory: Path) -> None:
        os.makedirs(str(directory), exist_ok=True)
        self.q_network.save_weights(os.path.join(str(directory), "q_network"))
        self.target_q_network.save_weights(
            os.path.join(str(directory), "target_q_network")
        )

    def load_checkpoint(self, directory: Path) -> None:
        self.q_network.load_weights(os.path.join(str(directory), "q_network"))
        self.target_q_network.load_weights(
            os.path.join(str(directory), "target_q_network")
        )

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

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
        parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
        parser.add_argument("--hide_task_id", type=str2bool, default=False)
        parser.add_argument("--buffer_type", type=str, default="fifo",
                            choices=[bt.value for bt in BufferType])
        parser.add_argument("--reset_buffer_on_task_change", type=str2bool, default=True)
        parser.add_argument("--reset_optimizer_on_task_change", type=str2bool, default=False)
        parser.add_argument("--reset_network_on_task_change", type=str2bool, default=False)
        parser.add_argument("--log_every", type=sci2int, default=1000)
        parser.add_argument("--save_freq_epochs", type=int, default=25)
        parser.add_argument(
            "--render_every", type=int, default=0,
            help="Play a greedy episode every N training episodes (0=disabled).",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace, env, logger, scene_ids: list) -> "DQN":
        from mariha.utils.running import get_readable_timestamp

        experiment_dir = Path(getattr(args, "experiment_dir", "experiments"))
        return cls(
            env=env,
            logger=logger,
            scene_ids=scene_ids,
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
            cl_hook_min_transitions=getattr(args, "cl_hook_min_transitions", 500),
            reset_buffer_on_task_change=getattr(args, "reset_buffer_on_task_change", True),
            reset_optimizer_on_task_change=getattr(args, "reset_optimizer_on_task_change", False),
            reset_network_on_task_change=getattr(args, "reset_network_on_task_change", False),
            hidden_sizes=tuple(getattr(args, "hidden_sizes", [256, 256])),
            hide_task_id=getattr(args, "hide_task_id", False),
            log_every=getattr(args, "log_every", 1000),
            save_freq_epochs=getattr(args, "save_freq_epochs", 25),
            render_every=getattr(args, "render_every", 0),
            post_burn_in_update_after=getattr(args, "post_burn_in_update_after", 0),
            experiment_dir=experiment_dir,
            timestamp=get_readable_timestamp(),
        )
