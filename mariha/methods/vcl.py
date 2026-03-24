"""Variational Continual Learning (VCL).

Replaces the standard dense layers in the actor (and optionally critic)
trunk with Bayesian dense layers whose weights are distributions
N(μ, σ²) rather than point estimates.

At each task boundary the variational posterior becomes the new prior for
the next task.  The auxiliary loss is the KL divergence between the current
posterior and the prior, which prevents catastrophic forgetting.

The CNN head is kept deterministic; only the dense trunk uses Bayesian
weights (matching the common practice in VCL papers for RL).

Reference: Nguyen et al., 2018 — arXiv:1710.10628.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import gymnasium
import tensorflow as tf

try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

from mariha.rl.models import (
    Input,
    Model,
    Sequential,
    Dense,
    Activation,
    Concatenate,
    build_conv_head,
    _choose_head,
)
from mariha.rl.sac import SAC


# ---------------------------------------------------------------------------
# Bayesian dense layer
# ---------------------------------------------------------------------------


class BayesianDense(_keras.layers.Layer):
    """Dense layer with variational weights N(μ, σ²).

    During training (``call(x, training=True)``) weights are sampled via the
    reparameterisation trick.  At inference time the mean μ is used.

    Args:
        units: Output dimensionality.
        log_sigma_init: Initial value for log(σ) of the posterior and prior.
        **kwargs: Forwarded to ``tf.keras.layers.Layer``.
    """

    def __init__(
        self,
        units: int,
        log_sigma_init: float = -3.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.log_sigma_init = log_sigma_init

    def build(self, input_shape: Tuple) -> None:
        fan_in = int(input_shape[-1])
        init_const = tf.initializers.Constant(self.log_sigma_init)

        # ---- posterior (trainable) ----
        self.mu_w = self.add_weight(
            shape=(fan_in, self.units), initializer="glorot_uniform", name="mu_w"
        )
        self.log_sigma_w = self.add_weight(
            shape=(fan_in, self.units), initializer=init_const, name="log_sigma_w"
        )
        self.mu_b = self.add_weight(
            shape=(self.units,), initializer="zeros", name="mu_b"
        )
        self.log_sigma_b = self.add_weight(
            shape=(self.units,), initializer=init_const, name="log_sigma_b"
        )

        # ---- prior (not trainable) ----
        self.prior_mu_w = self.add_weight(
            shape=(fan_in, self.units),
            trainable=False,
            initializer="zeros",
            name="prior_mu_w",
        )
        self.prior_log_sigma_w = self.add_weight(
            shape=(fan_in, self.units),
            trainable=False,
            initializer=init_const,
            name="prior_log_sigma_w",
        )
        self.prior_mu_b = self.add_weight(
            shape=(self.units,),
            trainable=False,
            initializer="zeros",
            name="prior_mu_b",
        )
        self.prior_log_sigma_b = self.add_weight(
            shape=(self.units,),
            trainable=False,
            initializer=init_const,
            name="prior_log_sigma_b",
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if training:
            sigma_w = tf.exp(self.log_sigma_w)
            sigma_b = tf.exp(self.log_sigma_b)
            w = self.mu_w + sigma_w * tf.random.normal(tf.shape(self.mu_w))
            b = self.mu_b + sigma_b * tf.random.normal(tf.shape(self.mu_b))
        else:
            w, b = self.mu_w, self.mu_b
        return x @ w + b

    def kl_divergence(self) -> tf.Tensor:
        """KL(posterior ‖ prior) summed over all weight and bias elements."""

        def _kl(mu, log_sigma, prior_mu, prior_log_sigma):
            sigma_sq = tf.exp(2.0 * log_sigma)
            prior_sigma_sq = tf.exp(2.0 * prior_log_sigma)
            return 0.5 * tf.reduce_sum(
                2.0 * (prior_log_sigma - log_sigma)
                + (sigma_sq + (mu - prior_mu) ** 2) / prior_sigma_sq
                - 1.0
            )

        return _kl(
            self.mu_w, self.log_sigma_w, self.prior_mu_w, self.prior_log_sigma_w
        ) + _kl(
            self.mu_b, self.log_sigma_b, self.prior_mu_b, self.prior_log_sigma_b
        )

    def update_prior(self) -> None:
        """Set prior ← current posterior (call at task boundaries)."""
        self.prior_mu_w.assign(self.mu_w)
        self.prior_log_sigma_w.assign(self.log_sigma_w)
        self.prior_mu_b.assign(self.mu_b)
        self.prior_log_sigma_b.assign(self.log_sigma_b)


# ---------------------------------------------------------------------------
# Variational MLP trunk (mirrors models.mlp but with BayesianDense)
# ---------------------------------------------------------------------------


def variational_mlp(
    state_shape: Tuple[int, ...],
    num_tasks: int,
    hidden_sizes: Iterable[int],
    activation: Callable,
    log_sigma_init: float = -3.0,
) -> Model:
    """Atari CNN + variational dense trunk.

    Mirrors :func:`~mariha.rl.models.mlp` but replaces the hidden ``Dense``
    layers with :class:`BayesianDense`.  The CNN head is kept deterministic.

    Args:
        state_shape: Observation shape, e.g. ``(84, 84, 4)``.
        num_tasks: Length of the one-hot task vector.
        hidden_sizes: Width of each hidden BayesianDense layer.
        activation: Activation function applied after each hidden layer.
        log_sigma_init: Initial log-σ for posterior and prior.

    Returns:
        A Keras Model with inputs ``[obs, task_id]`` and output size
        ``hidden_sizes[-1]``.
    """
    task_input = Input(shape=(num_tasks,), name="task_input", dtype=tf.float32)
    conv_in = Input(shape=state_shape, name="conv_head_in")
    conv_head = build_conv_head(conv_in)

    x = Concatenate()([conv_head, task_input])
    for size in hidden_sizes:
        x = BayesianDense(size, log_sigma_init=log_sigma_init)(x)
        x = Activation(activation)(x)

    return Model(inputs=[conv_in, task_input], outputs=x)


# ---------------------------------------------------------------------------
# VCL actor and critic
# ---------------------------------------------------------------------------


class VclMlpActor(_keras.Model):
    """Stochastic actor with a variational dense trunk.

    Interface matches :class:`~mariha.rl.models.MlpActor`.

    Args:
        state_space: Observation space.
        action_space: Discrete action space.
        num_tasks: Number of tasks (one-hot dimension).
        hidden_sizes: BayesianDense layer widths.
        activation: Hidden-layer activation.
        num_heads: Number of output heads.
        hide_task_id: Do not concatenate task one-hot (unused — kept for
            interface compatibility).
        log_sigma_init: Initial log-σ for BayesianDense.
    """

    def __init__(
        self,
        state_space: gymnasium.spaces.Box,
        action_space: gymnasium.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
        log_sigma_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.core = variational_mlp(
            state_space.shape, num_tasks, hidden_sizes, activation, log_sigma_init
        )
        self.head_mu = Sequential(
            [Input(shape=(list(hidden_sizes)[-1],)), Dense(action_space.n * num_heads)]
        )
        self.action_space = action_space

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        # Always pass training=True so the posterior is sampled.
        features = self.core([obs, one_hot_task_id], training=True)
        mu = self.head_mu(features)
        if self.num_heads > 1:
            mu = _choose_head(mu, self.num_heads, one_hot_task_id)
        return mu

    @property
    def common_variables(self) -> List[tf.Variable]:
        if self.num_heads > 1:
            return self.core.trainable_variables
        return self.core.trainable_variables + self.head_mu.trainable_variables


class VclMlpCritic(_keras.Model):
    """Q-value network with a variational dense trunk.

    Interface matches :class:`~mariha.rl.models.MlpCritic`.
    """

    def __init__(
        self,
        state_space: gymnasium.spaces.Box,
        action_space: gymnasium.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
        log_sigma_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.core = variational_mlp(
            state_space.shape, num_tasks, hidden_sizes, activation, log_sigma_init
        )
        self.head = Sequential(
            [Input(shape=(list(hidden_sizes)[-1],)), Dense(action_space.n * num_heads)]
        )

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        features = self.core([obs, one_hot_task_id], training=True)
        value = self.head(features)
        if self.num_heads > 1:
            value = _choose_head(value, self.num_heads, one_hot_task_id)
        return value

    @property
    def common_variables(self) -> List[tf.Variable]:
        if self.num_heads > 1:
            return self.core.trainable_variables
        return self.core.trainable_variables + self.head.trainable_variables


# ---------------------------------------------------------------------------
# VCL_SAC
# ---------------------------------------------------------------------------


class VCL_SAC(SAC):
    """Variational Continual Learning for SAC.

    Uses :class:`VclMlpActor` (and optionally :class:`VclMlpCritic`) as the
    policy networks.  The KL divergence between the current posterior and
    the task-start prior is added to the loss as the regularisation term.

    Args:
        cl_reg_coef: KL regularisation coefficient λ.
        bayesian_critic: If ``True``, the critic also uses Bayesian layers.
        log_sigma_init: Initial log-σ for all BayesianDense layers.
        **kwargs: Forwarded to :class:`~mariha.rl.sac.SAC`.
    """

    def __init__(
        self,
        cl_reg_coef: float = 1.0,
        bayesian_critic: bool = False,
        log_sigma_init: float = -3.0,
        **kwargs,
    ) -> None:
        from functools import partial
        from mariha.rl.models import MlpCritic

        # Use functools.partial to bind log_sigma_init without adding it to
        # policy_kwargs (which would break MlpCritic when bayesian_critic=False).
        kwargs.setdefault("actor_cl", partial(VclMlpActor, log_sigma_init=log_sigma_init))
        if bayesian_critic:
            kwargs.setdefault(
                "critic_cl", partial(VclMlpCritic, log_sigma_init=log_sigma_init)
            )
        else:
            kwargs.setdefault("critic_cl", MlpCritic)

        super().__init__(**kwargs)
        self.cl_reg_coef = cl_reg_coef
        self.bayesian_critic = bayesian_critic
        self._vcl_layers = self._collect_vcl_layers()

    # ------------------------------------------------------------------
    # Collect BayesianDense layers for KL computation
    # ------------------------------------------------------------------

    def _collect_vcl_layers(self) -> List[BayesianDense]:
        layers: List[BayesianDense] = []
        for layer in self.actor.core.layers:
            if isinstance(layer, BayesianDense):
                layers.append(layer)
        if self.bayesian_critic:
            for critic in (self.critic1, self.critic2):
                for layer in critic.core.layers:
                    if isinstance(layer, BayesianDense):
                        layers.append(layer)
        return layers

    # ------------------------------------------------------------------
    # SAC extension points
    # ------------------------------------------------------------------

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        """KL(posterior ‖ prior), active from the second task onward."""
        kl = tf.zeros([])
        for layer in self._vcl_layers:
            kl = kl + layer.kl_divergence()
        coef = tf.cond(seq_idx > 0, lambda: self.cl_reg_coef, lambda: 0.0)
        return kl * coef

    def on_task_start(self, current_task_idx: int) -> None:
        super().on_task_start(current_task_idx)
        if current_task_idx > 0:
            for layer in self._vcl_layers:
                layer.update_prior()
