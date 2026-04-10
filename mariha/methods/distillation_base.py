"""Shared base for distillation continual learning methods (DER, ClonEx).

:class:`DistillationMethod` factors out the bookkeeping that DER and
ClonEx share so each concrete method only has to override the
hyperparameters and (for ClonEx) opt into critic-side distillation.

Mechanics
---------
At every task boundary the regularizer samples ``episodic_mem_per_task``
transitions from the just-finished task's replay buffer, calls
``agent.distill_targets(obs, one_hot)``, and stores the resulting tensors
together with the raw ``(obs, one_hot)`` in an internal episodic memory.

At every gradient step :meth:`get_episodic_batch` returns a mini-batch of
old transitions, which the agent forwards to :meth:`adjust_gradients`.
That method computes the distillation gradients on the *episodic* batch
(not the live training batch) and adds them element-wise to the per-group
gradients the agent has already produced.

Per-agent dispatch
------------------
Each agent's ``distill_targets`` returns a *dict* of named tensors.
:class:`DistillationMethod` inspects the keys to decide which loss
formula to apply:

* ``"actor_logits"`` — softmax KL anchoring the stored vs. current
  policy logits.  Used by SAC and PPO.
* ``"q_values"`` — soft KL with temperature ``T`` (Rusu et al. 2016,
  *Policy Distillation*, arXiv:1511.06295).  Both stored and current
  Q-values are converted to a probability distribution via
  ``softmax(Q/T)``; the loss is ``T^2 · KL`` (the ``T^2`` factor
  preserves the gradient magnitude when ``T`` is varied).  Used by DQN.
* ``"critic1_q"`` / ``"critic2_q"`` — MSE on stored vs. current Q-value
  vectors.  Only added when :attr:`clone_critic` is True (ClonEx on
  SAC).  Each contributes to the corresponding ``critic1`` /
  ``critic2`` gradient group.
* ``"value"`` — MSE on the scalar PPO value head.  Only added when
  :attr:`clone_critic` is True.  Contributes to the ``value`` gradient
  group.

Subclasses
----------
:class:`DistillationMethod` is the base for the actor-only DER variant
out of the box; ClonEx is a thin subclass that flips
:attr:`clone_critic` on by default and exposes the additional
``critic_alpha`` hyperparameter.

The class is fully agent-agnostic: it never imports from
:mod:`mariha.rl.sac` etc. and all interactions with the agent go
through :meth:`BaseAgent.get_named_parameter_groups` and
:meth:`BaseAgent.distill_targets`.
"""

from __future__ import annotations

import argparse
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np
import tensorflow as tf

from mariha.methods.base import CLMethod

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------


class _DistillationMemory:
    """Fixed-capacity storage for distillation episodic memory.

    Holds observations, one-hot task IDs, and an arbitrary set of named
    target tensors derived from a snapshot of the agent at task end.
    Only used internally by :class:`DistillationMethod`.

    Args:
        obs_shape: Shape of a single observation.
        num_tasks: Length of the one-hot task vector.
        size: Maximum number of transitions to store.
        target_specs: Mapping ``key → (shape, dtype)`` for the per-key
            target buffers.  ``shape`` is the *per-transition* shape
            (without the leading time dimension); pass ``()`` for a
            scalar target.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_tasks: int,
        size: int,
        target_specs: Mapping[str, Tuple[Tuple[int, ...], np.dtype]],
    ) -> None:
        self.max_size = int(size)
        self.size = 0
        # Observations stored as float32 — distillation memory is
        # typically much smaller than the main replay buffer so the
        # 4x memory cost of float32 vs uint8 is acceptable in exchange
        # for a simpler storage path.
        self.obs_buf = np.zeros((self.max_size, *obs_shape), dtype=np.float32)
        self.one_hot_buf = np.zeros((self.max_size, num_tasks), dtype=np.float32)
        self.target_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = dict(
            target_specs
        )
        self._target_bufs: Dict[str, np.ndarray] = {
            key: np.zeros((self.max_size, *shape), dtype=dtype)
            for key, (shape, dtype) in self.target_specs.items()
        }

    def store_multiple(
        self,
        *,
        obs: np.ndarray,
        one_hot: np.ndarray,
        targets: Mapping[str, np.ndarray],
    ) -> None:
        """Append a batch of transitions, asserting capacity is not exceeded."""
        n = len(obs)
        if self.size + n > self.max_size:
            raise ValueError(
                f"_DistillationMemory: storing {n} would exceed capacity "
                f"{self.max_size} (current size {self.size})."
            )
        s, e = self.size, self.size + n
        self.obs_buf[s:e] = obs
        self.one_hot_buf[s:e] = one_hot
        for key in self.target_specs:
            if key not in targets:
                raise KeyError(
                    f"_DistillationMemory: missing target '{key}' in "
                    f"store_multiple call."
                )
            self._target_bufs[key][s:e] = targets[key]
        self.size += n

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        """Uniform sample (without replacement) from the stored data."""
        batch_size = min(batch_size, self.size)
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch: Dict[str, tf.Tensor] = {
            "obs": tf.convert_to_tensor(self.obs_buf[idxs]),
            "one_hot": tf.convert_to_tensor(self.one_hot_buf[idxs]),
        }
        for key, buf in self._target_bufs.items():
            batch[key] = tf.convert_to_tensor(buf[idxs])
        return batch


# ---------------------------------------------------------------------------
# DistillationMethod base
# ---------------------------------------------------------------------------


# Keys recognized in :meth:`BaseAgent.distill_targets` output.
_ACTOR_KEY = "actor_logits"
_QVAL_KEY = "q_values"
_CRITIC1_KEY = "critic1_q"
_CRITIC2_KEY = "critic2_q"
_VALUE_KEY = "value"


class DistillationMethod(CLMethod):
    """Base for DER / ClonEx-style episodic-memory distillation.

    Args:
        episodic_mem_per_task: Transitions stored per completed task.
            Total memory capacity is
            ``episodic_mem_per_task * agent.num_tasks``.
        episodic_batch_size: Mini-batch sampled from episodic memory at
            each gradient step.
        actor_alpha: Weight of the actor / Q-value distillation term.
            Multiplied into the gradient that gets added to the
            ``actor`` (SAC), ``policy`` (PPO), or ``q`` (DQN) group.
        critic_alpha: Weight of the critic-side distillation term.
            Multiplied into the gradient that gets added to the
            ``critic1`` / ``critic2`` (SAC) or ``value`` (PPO) groups.
            Only used when :attr:`clone_critic` is True.
        clone_critic: If True, also distil critic-side targets.  DER
            sets this to False (actor distillation only); ClonEx sets
            it to True.  On DQN this flag is ignored — the single
            ``q_values`` target already does both jobs.
        temperature: Temperature ``T`` for the soft-KL on
            ``q_values`` (Rusu 2016).  Higher temperatures produce
            softer target distributions.  Ignored when the agent
            distils ``actor_logits`` instead.
    """

    #: Subclasses override.  Used in CLI flag generation and log lines.
    name: str = "distill"

    def __init__(
        self,
        episodic_mem_per_task: int = 1000,
        episodic_batch_size: int = 128,
        actor_alpha: float = 0.1,
        critic_alpha: float = 0.1,
        clone_critic: bool = False,
        temperature: float = 2.0,
    ) -> None:
        self.episodic_mem_per_task = int(episodic_mem_per_task)
        self.episodic_batch_size = int(episodic_batch_size)
        self.actor_alpha = float(actor_alpha)
        self.critic_alpha = float(critic_alpha)
        self.clone_critic = bool(clone_critic)
        self.temperature = float(temperature)

        # Lazily allocated on first :meth:`_lazy_init` call so we can
        # query the agent for shape and key information.
        self._memory: Optional[_DistillationMemory] = None
        # Names of the parameter groups whose gradients receive a
        # distillation contribution.  Resolved during ``_lazy_init``.
        self._actor_group: Optional[str] = None
        self._critic_groups: List[str] = []
        # Recognized target keys present on this agent.
        self._has_actor_logits: bool = False
        self._has_q_values: bool = False
        self._has_critic_qs: bool = False
        self._has_value: bool = False

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _lazy_init(self, agent: "BaseAgent") -> None:
        """Allocate the episodic memory and resolve group names."""
        if self._memory is not None:
            return

        # Probe the agent for the available distillation target keys
        # and their shapes.  We use a tiny dummy batch (one transition)
        # so the call is cheap and side-effect-free.
        obs_shape = agent.obs_shape
        num_tasks = agent.num_tasks
        dummy_obs = tf.zeros((1, *obs_shape), dtype=tf.float32)
        dummy_one_hot = tf.zeros((1, num_tasks), dtype=tf.float32)
        targets = agent.distill_targets(dummy_obs, dummy_one_hot)

        self._has_actor_logits = _ACTOR_KEY in targets
        self._has_q_values = _QVAL_KEY in targets
        self._has_critic_qs = (
            _CRITIC1_KEY in targets and _CRITIC2_KEY in targets
        )
        self._has_value = _VALUE_KEY in targets

        if not (self._has_actor_logits or self._has_q_values):
            raise ValueError(
                f"{type(self).__name__}: agent.distill_targets() must return "
                f"at least one of '{_ACTOR_KEY}' or '{_QVAL_KEY}'.  "
                f"Got keys: {sorted(targets.keys())}"
            )

        # Resolve which named parameter group receives the actor /
        # Q-value distillation gradient.
        groups = agent.get_named_parameter_groups()
        for candidate in ("actor", "policy", "q"):
            if candidate in groups:
                self._actor_group = candidate
                break
        if self._actor_group is None:
            raise ValueError(
                f"{type(self).__name__}: no actor-equivalent group "
                f"('actor', 'policy', or 'q') found on agent.  "
                f"Available: {sorted(groups.keys())}"
            )

        # Resolve which groups receive critic distillation gradients.
        if self.clone_critic:
            if self._has_critic_qs:
                # SAC: distil to critic1 + critic2.
                if "critic1" in groups and "critic2" in groups:
                    self._critic_groups = ["critic1", "critic2"]
            elif self._has_value:
                # PPO: distil to the value head.
                if "value" in groups:
                    self._critic_groups = ["value"]

        # Build the target_specs from the dummy probe.  Each target
        # tensor's shape (excluding the batch dim) and dtype are
        # captured into the storage spec.
        target_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        if self._has_actor_logits:
            t = targets[_ACTOR_KEY]
            target_specs[_ACTOR_KEY] = (tuple(t.shape[1:]), np.float32)
        if self._has_q_values:
            t = targets[_QVAL_KEY]
            target_specs[_QVAL_KEY] = (tuple(t.shape[1:]), np.float32)
        if self.clone_critic and self._has_critic_qs:
            t1 = targets[_CRITIC1_KEY]
            t2 = targets[_CRITIC2_KEY]
            target_specs[_CRITIC1_KEY] = (tuple(t1.shape[1:]), np.float32)
            target_specs[_CRITIC2_KEY] = (tuple(t2.shape[1:]), np.float32)
        if self.clone_critic and self._has_value:
            t = targets[_VALUE_KEY]
            target_specs[_VALUE_KEY] = (tuple(t.shape[1:]), np.float32)

        self._memory = _DistillationMemory(
            obs_shape=obs_shape,
            num_tasks=num_tasks,
            size=self.episodic_mem_per_task * num_tasks,
            target_specs=target_specs,
        )

    # ------------------------------------------------------------------
    # CLMethod hooks
    # ------------------------------------------------------------------

    def on_task_end(self, agent: "BaseAgent", task_idx: int) -> None:
        """Snapshot ``episodic_mem_per_task`` transitions into episodic memory.

        Fires *before* the agent's replay buffer is reset, so the
        sampled batch belongs to the just-finished task.  No-op if the
        buffer hasn't accumulated enough data yet (e.g. very early
        burn-in finishing without producing real samples).
        """
        self._lazy_init(agent)
        buf = getattr(agent, "replay_buffer", None)
        if buf is None or buf.size < self.episodic_mem_per_task:
            return
        batch = buf.sample_batch(self.episodic_mem_per_task)
        obs = batch["obs"]
        one_hot = batch["one_hot"]
        target_tensors = agent.distill_targets(obs, one_hot)

        # Subset to the keys we actually store, convert to numpy.
        targets_np: Dict[str, np.ndarray] = {}
        for key in self._memory.target_specs:
            targets_np[key] = target_tensors[key].numpy()

        self._memory.store_multiple(
            obs=obs.numpy(),
            one_hot=one_hot.numpy(),
            targets=targets_np,
        )

    def get_episodic_batch(
        self, agent: "BaseAgent", *, task_idx: int
    ) -> Optional[Dict[str, tf.Tensor]]:
        """Return one mini-batch from episodic memory or ``None``.

        Returns ``None`` on the first task (no past data) or before
        :meth:`_lazy_init` has been triggered (e.g. before any task
        boundary has fired).  The agent forwards the returned batch
        verbatim to :meth:`adjust_gradients` later in the same step.
        """
        if task_idx == 0 or self._memory is None or self._memory.size == 0:
            return None
        return self._memory.sample_batch(self.episodic_batch_size)

    def adjust_gradients(
        self,
        agent: "BaseAgent",
        grads_by_group: Dict[str, List[Optional[tf.Tensor]]],
        *,
        task_idx: int,
        episodic_batch: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Dict[str, List[Optional[tf.Tensor]]]:
        """Add the distillation gradient to the appropriate groups.

        On the first task or when no episodic batch is available, this
        is a no-op (the original gradients pass through unchanged).
        Otherwise, computes the distillation loss on the episodic
        batch, takes the gradient w.r.t. the affected parameter
        groups, and adds it element-wise into ``grads_by_group``.
        """
        if task_idx == 0 or episodic_batch is None:
            return grads_by_group

        # Compute the actor-side distillation gradient.
        actor_grads = self._compute_actor_grads(agent, episodic_batch)
        grads_by_group[self._actor_group] = self._add_grads(
            grads_by_group[self._actor_group], actor_grads
        )

        # Optionally compute the critic-side distillation gradient.
        if self.clone_critic and self._critic_groups:
            critic_grads_by_group = self._compute_critic_grads(
                agent, episodic_batch
            )
            for gn, cg in critic_grads_by_group.items():
                grads_by_group[gn] = self._add_grads(grads_by_group[gn], cg)

        return grads_by_group

    # ------------------------------------------------------------------
    # Distillation gradient computation
    # ------------------------------------------------------------------

    def _compute_actor_grads(
        self,
        agent: "BaseAgent",
        episodic_batch: Dict[str, tf.Tensor],
    ) -> List[Optional[tf.Tensor]]:
        """Compute distillation gradient w.r.t. the actor-equivalent group.

        Dispatches on which target key is present:

        * ``actor_logits`` (SAC/PPO) → softmax KL.
        * ``q_values`` (DQN) → soft KL with temperature ``T``.
        """
        groups = agent.get_named_parameter_groups()
        actor_vars = groups[self._actor_group]

        with tf.GradientTape() as tape:
            tape.watch(actor_vars)
            current = agent.distill_targets(
                episodic_batch["obs"], episodic_batch["one_hot"]
            )
            if self._has_actor_logits:
                stored = episodic_batch[_ACTOR_KEY]
                current_logits = current[_ACTOR_KEY]
                p = tf.nn.softmax(stored)
                kl = tf.reduce_mean(
                    tf.reduce_sum(
                        p
                        * (
                            tf.nn.log_softmax(stored)
                            - tf.nn.log_softmax(current_logits)
                        ),
                        axis=-1,
                    )
                )
                loss = self.actor_alpha * kl
            else:
                # Q-value soft KL (Rusu 2016).
                stored = episodic_batch[_QVAL_KEY]
                current_q = current[_QVAL_KEY]
                T = self.temperature
                p = tf.nn.softmax(stored / T)
                kl = tf.reduce_mean(
                    tf.reduce_sum(
                        p
                        * (
                            tf.nn.log_softmax(stored / T)
                            - tf.nn.log_softmax(current_q / T)
                        ),
                        axis=-1,
                    )
                )
                # T^2 factor preserves the gradient magnitude as T varies.
                loss = self.actor_alpha * (T * T) * kl

        return tape.gradient(loss, actor_vars)

    def _compute_critic_grads(
        self,
        agent: "BaseAgent",
        episodic_batch: Dict[str, tf.Tensor],
    ) -> Dict[str, List[Optional[tf.Tensor]]]:
        """Compute MSE distillation gradients for the critic-equivalent groups."""
        groups = agent.get_named_parameter_groups()
        result: Dict[str, List[Optional[tf.Tensor]]] = {}

        for gn in self._critic_groups:
            critic_vars = groups[gn]
            with tf.GradientTape() as tape:
                tape.watch(critic_vars)
                current = agent.distill_targets(
                    episodic_batch["obs"], episodic_batch["one_hot"]
                )
                if gn == "critic1":
                    target = tf.stop_gradient(episodic_batch[_CRITIC1_KEY])
                    pred = current[_CRITIC1_KEY]
                elif gn == "critic2":
                    target = tf.stop_gradient(episodic_batch[_CRITIC2_KEY])
                    pred = current[_CRITIC2_KEY]
                elif gn == "value":
                    target = tf.stop_gradient(episodic_batch[_VALUE_KEY])
                    pred = current[_VALUE_KEY]
                else:
                    raise ValueError(
                        f"DistillationMethod: unsupported critic group "
                        f"'{gn}' — extend _compute_critic_grads()."
                    )
                mse = tf.reduce_mean((pred - target) ** 2)
                loss = self.critic_alpha * mse
            result[gn] = tape.gradient(loss, critic_vars)

        return result

    @staticmethod
    def _add_grads(
        base: List[Optional[tf.Tensor]],
        extra: List[Optional[tf.Tensor]],
    ) -> List[Optional[tf.Tensor]]:
        """Element-wise add ``extra`` into ``base``, tolerating ``None`` slots."""
        out: List[Optional[tf.Tensor]] = []
        for b, e in zip(base, extra):
            if b is None and e is None:
                out.append(None)
            elif b is None:
                out.append(e)
            elif e is None:
                out.append(b)
            else:
                out.append(b + e)
        return out

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add the shared distillation flags."""
        parser.add_argument(
            "--episodic_mem_per_task",
            type=int,
            default=1000,
            help="Transitions stored per completed task.",
        )
        parser.add_argument(
            "--episodic_batch_size",
            type=int,
            default=128,
            help="Mini-batch size sampled from episodic memory.",
        )
        parser.add_argument(
            "--actor_alpha",
            type=float,
            default=0.1,
            help="Weight of the actor / Q-value distillation term.",
        )
        parser.add_argument(
            "--critic_alpha",
            type=float,
            default=0.1,
            help="Weight of the critic distillation term (ClonEx only).",
        )
        parser.add_argument(
            "--distill_temperature",
            type=float,
            default=2.0,
            help="Temperature T for soft Q-value KL on DQN (Rusu 2016).",
        )

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        agent: "BaseAgent",
    ) -> "DistillationMethod":
        return cls(
            episodic_mem_per_task=getattr(args, "episodic_mem_per_task", 1000),
            episodic_batch_size=getattr(args, "episodic_batch_size", 128),
            actor_alpha=getattr(args, "actor_alpha", 0.1),
            critic_alpha=getattr(args, "critic_alpha", 0.1),
            clone_critic=False,
            temperature=getattr(args, "distill_temperature", 2.0),
        )
