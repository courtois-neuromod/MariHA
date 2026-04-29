"""Proximal Policy Optimization for the MariHA continual-learning benchmark.

On-policy actor-critic with a fixed-length rollout buffer and clipped
surrogate objective.  Architecture and hyperparameters are adapted from
the ``ppo_study`` reference implementation (PyTorch), translated to
TensorFlow / Keras.

Built on :class:`mariha.rl.base.BaseAgent` with
``update_granularity="per_rollout"``: ``should_update`` returns True
once the rollout buffer is full, at which point ``update_step``
bootstraps the final value, computes GAE returns/advantages, runs
``n_epochs`` of minibatch updates, and resets the buffer.

Usage::

    mariha-run-cl --agent ppo --subject sub-01 --seed 0 --lr 1e-4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

from mariha.replay.buffers import RolloutBuffer
from mariha.rl import models
from mariha.rl.base import BaseAgent, run_burn_in
from mariha.utils.logging import EpochLogger


class PPO(BaseAgent):
    """Proximal Policy Optimization for continual learning on MariHA.

    Per-rollout update style: collects ``rollout_length`` transitions
    into a :class:`RolloutBuffer`, then runs ``n_epochs`` of clipped
    minibatch updates over the rollout.  The shared training loop
    drives the per-step interaction; PPO's algorithm-specific work
    lives entirely inside :meth:`update_step`.
    """

    update_granularity: str = "per_rollout"

    def __init__(
        self,
        env,
        logger: EpochLogger,
        run_ids: List[str],
        *,
        seed: int = 0,
        # ---- PPO hyperparameters ----
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
        # ---- task-change behaviour ----
        reset_optimizer_on_task_change: bool = False,
        reset_network_on_task_change: bool = False,
        # ---- architecture ----
        hide_task_id: bool = False,
        # ---- BaseAgent / logging ----
        log_every: int = 1000,
        save_freq_epochs: int = 25,
        render_every: int = 0,
        experiment_dir: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            logger=logger,
            run_ids=run_ids,
            agent_name="ppo",
            seed=seed,
            log_every=log_every,
            save_freq_epochs=save_freq_epochs,
            render_every=render_every,
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )

        # ---- PPO hyperparameters ----
        self.rollout_length = int(rollout_length)
        self.n_epochs = int(n_epochs)
        self.n_minibatches = int(n_minibatches)
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_network_on_task_change = reset_network_on_task_change

        self.hide_task_id = hide_task_id

        # ---- networks + optimizer + rollout buffer ----
        self.model = models.PPOActorCritic(**self._policy_kwargs())
        self.optimizer = _keras.optimizers.legacy.Adam(learning_rate=lr)

        self.minibatch_size = self.rollout_length // self.n_minibatches
        self.rollout_buffer = RolloutBuffer(
            obs_shape=self.obs_shape,
            size=self.rollout_length,
            num_tasks=self.num_tasks,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Cached for the post-rollout bootstrap.  ``store_transition``
        # updates these on every step so that ``update_step`` can
        # compute V(_last_next_obs) for the GAE final term.
        self._last_next_obs: Optional[np.ndarray] = None
        self._last_one_hot: Optional[np.ndarray] = None

        # Per-task compiled gradient step.  Recompiled in
        # :meth:`on_task_change` so the embedded ``current_task_idx``
        # is captured as a Python int constant — that lets the CL
        # method's trace-time branches (e.g. "skip the penalty on
        # task 0") bake into the per-task graph.
        self._gradient_step_fn = self._make_gradient_step_fn(0)

    # ==================================================================
    # Algorithm internals
    # ==================================================================

    def _policy_kwargs(self) -> Dict[str, Any]:
        return dict(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            num_tasks=self.num_tasks,
            hide_task_id=self.hide_task_id,
        )

    @tf.function
    def _get_action_tf(
        self,
        obs: tf.Tensor,
        one_hot: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compiled action selection — returns (action, log_prob, value) as tensors."""
        logits, value = self.model(obs, one_hot)
        probs = tf.nn.softmax(logits[0])
        action = tf.random.categorical(tf.math.log(probs[tf.newaxis]), 1)[0, 0]
        action = tf.cast(action, tf.int32)
        log_prob = tf.math.log(probs[action] + 1e-8)
        return action, log_prob, value[0, 0]

    def _get_action_with_info(
        self, obs: np.ndarray, task_one_hot: np.ndarray
    ) -> Tuple[int, float, float]:
        """Return ``(action, log_prob, value)`` for rollout collection."""
        obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
        one_hot_t = tf.expand_dims(tf.cast(task_one_hot, tf.float32), 0)
        action, log_prob, value = self._get_action_tf(obs_t, one_hot_t)
        return int(action.numpy()), float(log_prob.numpy()), float(value.numpy())

    def _make_gradient_step_fn(self, current_task_idx: int):
        """Build a per-task compiled minibatch gradient step.

        ``current_task_idx`` is captured in the closure as a Python
        int.  This is the same per-task recompile pattern SAC uses for
        ``learn_on_batch`` — it lets ``cl_method.compute_loss_penalty``
        return ``tf.zeros([])`` on task 0 at trace time so the penalty
        is fully optimized away from the first-task graph.
        """

        @tf.function
        def gradient_step(
            obs,
            actions,
            returns,
            advantages,
            old_log_probs,
            one_hot,
            episodic_batch=None,
        ):
            with tf.GradientTape() as tape:
                logits, values = self.model(obs, one_hot)
                values = tf.squeeze(values, axis=-1)

                probs = tf.nn.softmax(logits)
                log_probs_all = tf.math.log(probs + 1e-8)
                new_log_probs = tf.reduce_sum(
                    log_probs_all * tf.one_hot(actions, self.act_dim),
                    axis=-1,
                )

                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                )
                actor_loss = -tf.reduce_mean(
                    tf.minimum(ratio * advantages, clipped_ratio * advantages)
                )

                critic_loss = tf.reduce_mean(
                    tf.keras.losses.huber(returns, values, delta=1.0)
                )

                entropy = -tf.reduce_sum(probs * log_probs_all, axis=-1)
                entropy_loss = -tf.reduce_mean(entropy)

                total_loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.ent_coef * entropy_loss
                )

                # CL regularization (e.g. EWC quadratic anchor).
                # ``compute_loss_penalty`` returns ``tf.zeros([])`` on
                # the first task, evaluated at trace time.
                if self.cl_method is not None:
                    total_loss = (
                        total_loss
                        + self.cl_method.compute_loss_penalty(
                            self, task_idx=current_task_idx
                        )
                    )

            grads = tape.gradient(total_loss, self.model.trainable_variables)

            # Hand the gradients to the CL method for in-place adjustment
            # (AGEM projection, PackNet masking, DER additive distillation
            # term, ...).  The single-key view matches PPO's optimizer
            # view exactly.
            grads_by_group: Dict[str, List[Optional[tf.Tensor]]] = {
                "policy": list(grads)
            }
            if self.cl_method is not None:
                grads_by_group = self.cl_method.adjust_gradients(
                    self,
                    grads_by_group,
                    task_idx=current_task_idx,
                    episodic_batch=episodic_batch,
                )
            grads = grads_by_group["policy"]

            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            # Refresh so ``after_gradient_step`` sees the gradients
            # actually applied (post-clip).
            grads_by_group = {"policy": list(grads)}

            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables)
            )

            metrics = dict(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=tf.reduce_mean(entropy),
                total_loss=total_loss,
            )

            if self.cl_method is not None:
                self.cl_method.after_gradient_step(
                    self,
                    grads_by_group,
                    task_idx=current_task_idx,
                    metrics=metrics,
                )

            return metrics

        return gradient_step

    def _do_minibatch_updates(self, current_task_idx: int) -> Dict[str, float]:
        """Run ``n_epochs`` of minibatch updates over the current rollout."""
        metrics: Dict[str, float] = {}
        episodic_batch = (
            self.cl_method.get_episodic_batch(
                self, task_idx=current_task_idx
            )
            if self.cl_method is not None
            else None
        )
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_batches(self.minibatch_size):
                results = self._gradient_step_fn(
                    batch["obs"],
                    batch["actions"],
                    batch["returns"],
                    batch["advantages"],
                    batch["old_log_probs"],
                    batch["one_hot"],
                    episodic_batch,
                )
                metrics = {k: float(v) for k, v in results.items()}
        return metrics

    # ==================================================================
    # Agent-agnostic CL contract (Phase 4)
    # ==================================================================

    def get_named_parameter_groups(self) -> Dict[str, List[tf.Variable]]:
        """Named parameter groups for agent-agnostic CL methods.

        PPO is a single shared-trunk actor-critic — backbone, actor
        head, and critic head are all updated by one optimizer step
        over ``model.trainable_variables``.  We expose a single
        ``"policy"`` group so :meth:`_gradient_step` can pass a
        non-overlapping ``grads_by_group`` view to
        ``cl_method.adjust_gradients`` that exactly matches the
        optimizer's view.

        ``ParameterRegularizer`` defaults to selecting ``"policy"`` (its
        default search order is ``actor → policy → q``), which anchors
        the entire actor-critic network — the right semantics for
        "preserve the old policy" since a clamp on the actor head alone
        would let the shared trunk drift freely.
        """
        return {"policy": list(self.model.trainable_variables)}

    def forward_for_importance(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Differentiable forward output per parameter group.

        Returns the policy logits for the ``"policy"`` group.  EWC and
        MAS take the Jacobian of these logits w.r.t. every parameter
        in ``model.trainable_variables`` — including the shared
        backbone, since logits flow through it.  Note that the value
        head's contribution to backbone importance is not measured;
        we treat the actor's gradient sensitivity as the canonical
        proxy for "the policy's reliance on each parameter".
        """
        logits, _ = self.model(obs, one_hot)
        return {"policy": logits}

    def distill_targets(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Distillation targets for DER/ClonEx-style CL methods.

        PPO distils both the policy logits (``actor_logits``) and the
        scalar value estimate (``value``).  ``DistillationMethod``
        dispatches on the keys: DER consumes only ``actor_logits``;
        ClonEx additionally consumes ``value``.
        """
        logits, value = self.model(obs, one_hot)
        return {
            "actor_logits": logits,
            "value": tf.squeeze(value, axis=-1),
        }

    def _compute_reference_loss(
        self, batch: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """PPO reference loss on raw transitions for AGEM-style projection.

        AGEM stores raw transitions ``(obs, action, reward, next_obs,
        done, one_hot)`` from past tasks, but PPO's standard loss
        needs ``returns`` and ``advantages`` from a rollout it owns.
        For the reference-gradient pass we fall back to the simplest
        meaningful objective: the negative log-likelihood of the
        stored actions under the *current* policy — i.e., a behaviour-
        cloning loss on past trajectories.

        Projecting the current PPO update onto the half-space defined
        by this loss keeps the policy from forgetting how to produce
        the actions it took on previous tasks, which is the spirit of
        AGEM applied to on-policy methods (Rolnick et al. 2019,
        *Experience Replay for Continual Learning*).
        """
        obs = batch["obs"]
        actions = batch["actions"]
        one_hot = batch["one_hot"]

        logits, _ = self.model(obs, one_hot)
        log_probs_all = tf.nn.log_softmax(logits)
        action_log_probs = tf.reduce_sum(
            log_probs_all * tf.one_hot(actions, self.act_dim), axis=-1
        )
        bc_loss = -tf.reduce_mean(action_log_probs)
        return {"policy": bc_loss}

    # ==================================================================
    # BenchmarkAgent contract
    # ==================================================================

    @tf.function
    def _get_action_deterministic_tf(
        self, obs: tf.Tensor, one_hot: tf.Tensor
    ) -> tf.Tensor:
        logits, _ = self.model(obs, one_hot)
        return tf.cast(tf.argmax(logits[0]), tf.int32)

    def get_action(
        self,
        obs: np.ndarray,
        task_one_hot: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        obs_t = tf.expand_dims(tf.cast(obs, tf.float32), 0)
        one_hot_t = tf.expand_dims(tf.cast(task_one_hot, tf.float32), 0)
        if deterministic:
            return int(self._get_action_deterministic_tf(obs_t, one_hot_t).numpy())
        action, _, _ = self._get_action_tf(obs_t, one_hot_t)
        return int(action.numpy())

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
        action, log_prob, value = self._get_action_with_info(obs, one_hot)
        return action, {"log_prob": log_prob, "value": value}

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
        done = terminated or truncated
        self.rollout_buffer.store(
            obs,
            action,
            reward,
            done,
            extras["value"],
            extras["log_prob"],
            one_hot,
        )
        # Cache the post-step state for the bootstrap.  These get
        # overwritten on every store_transition; by the time
        # ``update_step`` runs (immediately after the rollout buffer
        # fills) they hold the next-state of the very last step.
        self._last_next_obs = next_obs
        self._last_one_hot = one_hot

    def should_update(self, global_step: int) -> bool:
        return self.rollout_buffer.ptr >= self.rollout_length

    def update_step(
        self, *, global_step: int, current_task_idx: int
    ) -> None:
        if self.rollout_buffer.ptr == 0:
            return

        # Bootstrap V(s_T) for the GAE final term.  Note that this uses
        # the model's value estimate even on terminated transitions —
        # consistent with the legacy PPO loop and tolerated because the
        # bias is small relative to the GAE residual.
        obs_t = tf.expand_dims(
            tf.cast(self._last_next_obs, tf.float32), 0
        )
        one_hot_t = tf.expand_dims(
            tf.cast(self._last_one_hot, tf.float32), 0
        )
        _, last_value = self.model(obs_t, one_hot_t)
        self.rollout_buffer.compute_returns_and_advantages(
            float(last_value[0, 0].numpy())
        )

        t_update = time.time()
        metrics = self._do_minibatch_updates(current_task_idx)
        self.logger.log(
            f"PPO update | step {global_step} | "
            f"loss={metrics.get('total_loss', 0):.4f} | "
            f"{time.time() - t_update:.2f}s"
        )
        self.logger.store(
            {
                "train/loss_actor": metrics.get("actor_loss", 0.0),
                "train/loss_critic": metrics.get("critic_loss", 0.0),
                "train/entropy": metrics.get("entropy", 0.0),
            }
        )

        self.rollout_buffer.reset()

    def save_weights(self, directory: Path) -> None:
        directory = Path(directory)
        self.model.save_weights(str(directory / "ppo_actor_critic"))

    def load_weights(self, directory: Path) -> None:
        directory = Path(directory)
        self.model.load_weights(str(directory / "ppo_actor_critic"))

    # ==================================================================
    # Optional BaseAgent hooks
    # ==================================================================

    def on_task_start(
        self, current_task_idx: int, run_id: str = ""
    ) -> None:
        """PPO's task-start log line + forward to attached CL method."""
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
        """PPO's task-end log line + forward to attached CL method."""
        self.logger.log(f"Task end:   idx={current_task_idx}", color="white")
        super().on_task_end(current_task_idx)

    def on_task_change(self, new_task_idx: int, new_run_id: str) -> None:
        if self.reset_optimizer_on_task_change:
            from mariha.utils.running import reset_optimizer

            reset_optimizer(self.optimizer)
        if self.reset_network_on_task_change:
            from mariha.utils.running import reset_weights

            reset_weights(
                self.model, models.PPOActorCritic, self._policy_kwargs()
            )

        # Recompile so the embedded ``current_task_idx`` constant
        # matches the new task — see ``_make_gradient_step_fn``.
        self._gradient_step_fn = self._make_gradient_step_fn(new_task_idx)

    def get_log_tabular_keys(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("train/loss_actor", {"average_only": True}),
            ("train/loss_critic", {"average_only": True}),
            ("train/entropy", {"average_only": True}),
        ]

    # ==================================================================
    # Burn-in
    # ==================================================================

    def burn_in(self, burn_in_spec, num_steps: int) -> None:
        run_burn_in(self, burn_in_spec, num_steps)

    def on_burn_in_end(self) -> None:
        """Reset the rollout buffer after burn-in."""
        self.rollout_buffer.reset()
        self.logger.log("[burn-in] PPO: rollout buffer reset.", color="cyan")

    # ==================================================================
    # Config interface
    # ==================================================================

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
        parser.add_argument("--hide_task_id", type=str2bool, default=False)
        parser.add_argument(
            "--reset_optimizer_on_task_change", type=str2bool, default=False
        )
        parser.add_argument(
            "--reset_network_on_task_change", type=str2bool, default=False
        )
        parser.add_argument("--log_every", type=int, default=1000)
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
    ) -> "PPO":
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
            reset_optimizer_on_task_change=getattr(
                args, "reset_optimizer_on_task_change", False
            ),
            reset_network_on_task_change=getattr(
                args, "reset_network_on_task_change", False
            ),
            hide_task_id=getattr(args, "hide_task_id", False),
            experiment_dir=experiment_dir,
            timestamp=timestamp,
        )
