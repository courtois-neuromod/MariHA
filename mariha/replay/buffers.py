"""Experience replay buffers for SAC training.

Provides four buffer types:

- ``ReplayBuffer``             — FIFO ring buffer (default).
- ``ReservoirReplayBuffer``    — Reservoir sampling (uniform over all seen data).
- ``PrioritizedReplayBuffer``  — Priority via ``SumTree``.
- ``PrioritizedExperienceReplay`` — PER via ``SegmentTree`` (cleaner IS weights).
- ``EpisodicMemory``           — Fixed-capacity buffer that never overwrites
                                 (used by AGEM, DER++, ClonEx).

All buffers store transitions as ``(obs, action, reward, next_obs, done,
one_hot_task_id)``.  ``sample_batch()`` returns a dict of ``tf.Tensor``
objects ready for gradient computations.

Ported from COOM; only the import path for the tree helpers has changed.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from mariha.replay.tree import SumTree, SegmentTree


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"
    PRIORITY = "priority"
    PER = "per"


# ---------------------------------------------------------------------------
# FIFO replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Simple FIFO experience replay buffer.

    Args:
        obs_shape: Shape of a single observation (e.g. ``(84, 84, 4)``).
        size: Maximum number of transitions to store.
        num_tasks: Length of the one-hot task-ID vector.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        size: int,
        num_tasks: int,
    ) -> None:
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.actions_buf = np.zeros(size, dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.one_hot_buf = np.zeros([size, num_tasks], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        one_hot: np.ndarray,
    ) -> None:
        """Store a single transition."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.one_hot_buf[self.ptr] = one_hot
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        """Sample a random mini-batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs]),
        )


# ---------------------------------------------------------------------------
# Episodic memory (no overwriting — used by AGEM, DER++, ClonEx)
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """Fixed-capacity episodic buffer that does not overwrite old entries.

    Args:
        obs_shape: Shape of a single observation.
        act_dim: Number of discrete actions (used for target buffers).
        size: Maximum number of transitions.
        num_tasks: Length of the one-hot task-ID vector.
        save_targets: If ``True``, also store actor logits and Q-value
            predictions (required by DER++ / ClonEx distillation).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        act_dim: int,
        size: int,
        num_tasks: int,
        save_targets: bool = False,
    ) -> None:
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.actions_buf = np.zeros(size, dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.one_hot_buf = np.zeros([size, num_tasks], dtype=np.float32)
        self.size, self.max_size = 0, size
        self.save_targets = save_targets
        if self.save_targets:
            self.actor_logits_buf = np.zeros([size, act_dim], dtype=np.float32)
            self.critic1_pred_buf = np.zeros([size, act_dim], dtype=np.float32)
            self.critic2_pred_buf = np.zeros([size, act_dim], dtype=np.float32)

    def store_multiple(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        one_hot: np.ndarray,
        **kwargs: Dict[str, np.ndarray],
    ) -> None:
        """Append a batch of transitions (no overwrite — asserts capacity)."""
        n = len(obs)
        assert len(actions) == len(rewards) == len(next_obs) == len(done) == n
        assert self.size + n <= self.max_size
        s, e = self.size, self.size + n
        self.obs_buf[s:e] = obs
        self.next_obs_buf[s:e] = next_obs
        self.actions_buf[s:e] = actions
        self.rewards_buf[s:e] = rewards
        self.done_buf[s:e] = done
        self.one_hot_buf[s:e] = one_hot
        if self.save_targets:
            self.actor_logits_buf[s:e] = kwargs["actor_logits"]
            self.critic1_pred_buf[s:e] = kwargs["critic1_preds"]
            self.critic2_pred_buf[s:e] = kwargs["critic2_preds"]
        self.size += n

    def sample_batch(
        self,
        batch_size: int,
        task_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, tf.Tensor]:
        """Sample a mini-batch, optionally re-weighted by task frequency."""
        batch_size = min(batch_size, self.size)
        if task_weights is not None:
            task_ids = self.one_hot_buf[: self.size]
            example_weights = task_weights[task_ids]
            example_weights /= example_weights.sum()
            idxs = np.random.choice(self.size, size=batch_size, replace=False, p=example_weights)
        else:
            idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs]),
        )
        if self.save_targets:
            batch["actor_logits"] = tf.convert_to_tensor(self.actor_logits_buf[idxs])
            batch["critic1_preds"] = tf.convert_to_tensor(self.critic1_pred_buf[idxs])
            batch["critic2_preds"] = tf.convert_to_tensor(self.critic2_pred_buf[idxs])
        return batch


# ---------------------------------------------------------------------------
# Reservoir sampling buffer
# ---------------------------------------------------------------------------


class ReservoirReplayBuffer(ReplayBuffer):
    """FIFO buffer extended with reservoir sampling.

    Ensures uniform distribution over all transitions seen so far, not just
    the most recent ones.

    Args:
        obs_shape: Shape of a single observation.
        size: Buffer capacity.
        num_tasks: Length of the one-hot task-ID vector.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        size: int,
        num_tasks: int,
    ) -> None:
        super().__init__(obs_shape, size, num_tasks)
        self.timestep = 0

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        one_hot: np.ndarray,
    ) -> None:
        current_t = self.timestep
        self.timestep += 1
        if current_t < self.max_size:
            buffer_idx = current_t
        else:
            buffer_idx = random.randint(0, current_t)
            if buffer_idx >= self.max_size:
                return
        self.obs_buf[buffer_idx] = obs
        self.next_obs_buf[buffer_idx] = next_obs
        self.actions_buf[buffer_idx] = action
        self.rewards_buf[buffer_idx] = reward
        self.done_buf[buffer_idx] = done
        self.one_hot_buf[buffer_idx] = one_hot
        self.size = min(self.size + 1, self.max_size)


# ---------------------------------------------------------------------------
# Prioritized replay (SumTree)
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay using a ``SumTree``.

    Args:
        obs_shape: Shape of a single observation.
        size: Buffer capacity.
        num_tasks: Length of the one-hot task-ID vector.
    """

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment = 0.001
    absolute_error_upper = 1.0

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        size: int,
        num_tasks: int,
    ) -> None:
        super().__init__(obs_shape, size, num_tasks)
        self.buffer = SumTree(size)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        one_hot: np.ndarray,
    ) -> None:
        max_priority = np.max(self.buffer.tree[-self.buffer.capacity :])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.buffer.add(max_priority, (obs, next_obs, action, reward, done, one_hot))
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        memory_b = []
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_ISWeights = np.empty((batch_size, 1), dtype=np.float32)
        priority_segment = self.buffer.total_priority / batch_size
        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment])
        p_min = np.min(self.buffer.tree[-self.buffer.capacity :]) / self.buffer.total_priority
        max_weight = 1e-7 if p_min == 0 else (p_min * batch_size) ** (-self.PER_b)
        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.buffer.get_leaf(value)
            sampling_probabilities = priority / self.buffer.total_priority
            b_ISWeights[i, 0] = (
                np.power(batch_size * sampling_probabilities, -self.PER_b) / max_weight
            )
            b_idx[i] = index
            memory_b.append(data)
        memory_b = np.array(memory_b)
        return dict(
            obs=tf.convert_to_tensor(memory_b[:, 0].tolist(), dtype=tf.float32),
            next_obs=tf.convert_to_tensor(memory_b[:, 1].tolist(), dtype=tf.float32),
            actions=tf.convert_to_tensor(memory_b[:, 2].tolist(), dtype=tf.int32),
            rewards=tf.convert_to_tensor(memory_b[:, 3].tolist(), dtype=tf.float32),
            done=tf.convert_to_tensor(memory_b[:, 4].tolist(), dtype=tf.float32),
            one_hot=tf.convert_to_tensor(memory_b[:, 5].tolist(), dtype=tf.float32),
            idxs=tf.convert_to_tensor(b_idx, dtype=tf.int32),
            weights=tf.convert_to_tensor(b_ISWeights, dtype=tf.float32),
        )

    def update_weights(self, tree_idx: np.ndarray, abs_errors: tf.Tensor) -> None:
        """Update priorities in the SumTree after a gradient step."""
        abs_errors = np.array(abs_errors) + self.PER_e
        clipped = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.buffer.update(ti, p)


# ---------------------------------------------------------------------------
# Prioritized Experience Replay (SegmentTree, cleaner IS weights)
# ---------------------------------------------------------------------------


class PrioritizedExperienceReplay(ReplayBuffer):
    """PER using a ``SegmentTree`` for more stable importance-sampling weights.

    Reference: arXiv:1511.05952.

    Args:
        obs_shape: Shape of a single observation.
        size: Buffer capacity.
        num_tasks: Length of the one-hot task-ID vector.
        alpha: Prioritization exponent (0 = uniform, 1 = full priority).
        beta: Importance-sampling exponent.
        weight_norm: Whether to normalize IS weights by the max within the batch.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        size: int,
        num_tasks: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        weight_norm: bool = True,
    ) -> None:
        ReplayBuffer.__init__(self, obs_shape, size, num_tasks)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        self.absolute_error_upper = 1.0
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self._weight_norm = weight_norm

    def _init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio ** self._alpha

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        one_hot: np.ndarray,
    ) -> None:
        super().store(obs, action, reward, next_obs, done, one_hot)
        self._init_weight(self.ptr)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        scalar = np.random.rand(batch_size) * self.weight.reduce()
        idxs = self.weight.get_prefix_sum_idx(scalar)
        weight = self.get_weight(idxs)
        if self._weight_norm:
            weight = weight / np.max(weight)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs]),
            idxs=tf.convert_to_tensor(idxs),
            weights=tf.convert_to_tensor(weight),
        )

    def get_weight(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute importance-sampling weights for the given indices."""
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_weights(
        self, index: np.ndarray, new_weight: Union[np.ndarray, tf.Tensor]
    ) -> None:
        """Update priorities after a gradient step."""
        weight = np.abs(np.array(new_weight, dtype=np.float64)) + self.__eps
        self.weight[index] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def set_beta(self, beta: float) -> None:
        self._beta = beta
