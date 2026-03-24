"""Neural network architectures for SAC (actor and critic).

The architecture follows COOM's design: a shared convolutional feature
extractor (Atari-style CNN) concatenated with the one-hot task ID, followed
by two dense layers.

Classes:
    ``MlpActor``  — Stochastic discrete-action policy (returns action logits).
    ``MlpCritic`` — Q-value network (returns Q(s,a) for all discrete actions).

Both accept the pixel observation ``(H, W, C)`` and the task one-hot vector
as separate inputs, matching the COOM calling convention::

    logits = actor(obs_batch, one_hot_batch)
    q_values = critic(obs_batch, one_hot_batch)

Ported from COOM; VizDoom / LSTM / mrunner references removed.  The LSTM
option is preserved as a flag but not actively used in the MariHA benchmark.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import gymnasium
import tensorflow as tf

try:
    import tf_keras as _keras
except ImportError:
    import tensorflow.keras as _keras  # type: ignore[no-redef]

Input = _keras.Input
Model = _keras.Model
Sequential = _keras.Sequential
Activation = _keras.layers.Activation
Concatenate = _keras.layers.Concatenate
Conv2D = _keras.layers.Conv2D
Dense = _keras.layers.Dense
Flatten = _keras.layers.Flatten
LayerNormalization = _keras.layers.LayerNormalization
LSTM = _keras.layers.LSTM
TimeDistributed = _keras.layers.TimeDistributed


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------


def build_conv_head(conv_in: tf.Tensor, use_lstm: bool = False) -> tf.Tensor:
    """Build the Atari-style convolutional feature extractor.

    Architecture: 32×8×4 → 64×4×2 → 64×3×1 → Flatten (→ 3136 units).
    With ``use_lstm=True``, the spatial features are fed through an LSTM(512)
    instead of a simple Flatten.

    Args:
        conv_in: Keras tensor with shape ``(H, W, C)`` (input to the network).
        use_lstm: If ``True``, wrap each conv layer with ``TimeDistributed``
            and pool with LSTM.  Expects the leading dimension to be the time
            axis in that case.

    Returns:
        A Keras tensor with shape ``(batch, features)``.
    """
    for filters, kernel, stride in zip((32, 64, 64), (8, 4, 3), (4, 2, 1)):
        conv_layer = Conv2D(filters, kernel, stride, activation="relu")
        conv_in = (
            TimeDistributed(conv_layer)(conv_in) if use_lstm else conv_layer(conv_in)
        )
    if use_lstm:
        conv_in = TimeDistributed(Flatten())(conv_in)
        conv_in = LSTM(512, activation="tanh")(conv_in)
    else:
        conv_in = Flatten()(conv_in)
    return conv_in


def mlp(
    state_shape: Tuple[int, ...],
    num_tasks: int,
    hidden_sizes: Iterable[int],
    activation: Callable,
    use_layer_norm: bool = False,
    use_lstm: bool = False,
    hide_task_id: bool = False,
) -> Model:
    """Build a shared CNN + MLP trunk that accepts ``(obs, task_id)`` inputs.

    Args:
        state_shape: Observation shape, e.g. ``(84, 84, 4)``.
        num_tasks: Length of the one-hot task vector.
        hidden_sizes: Sequence of dense-layer widths after the CNN.
        activation: Activation function for hidden layers.
        use_layer_norm: If ``True``, apply ``LayerNormalization`` + ``tanh``
            after the first dense layer.
        use_lstm: Forward to ``build_conv_head``.
        hide_task_id: If ``True``, the task one-hot is not concatenated to
            the CNN features (useful for ablations).

    Returns:
        A Keras ``Model`` with inputs ``[obs, task_id]`` (or just ``obs``
        when ``hide_task_id=True``) and output of size ``hidden_sizes[-1]``.
    """
    task_input = Input(shape=(num_tasks,), name="task_input", dtype=tf.float32)
    conv_in = Input(shape=state_shape, name="conv_head_in")
    conv_head = build_conv_head(conv_in, use_lstm)

    model = conv_head if hide_task_id else Concatenate()([conv_head, task_input])
    model = Dense(hidden_sizes[0])(model)
    if use_layer_norm:
        model = LayerNormalization()(model)
        model = Activation(tf.nn.tanh)(model)
    else:
        model = Activation(activation)(model)
    for size in list(hidden_sizes)[1:]:
        model = Dense(size, activation=activation)(model)

    inputs = conv_in if hide_task_id else [conv_in, task_input]
    return Model(inputs=inputs, outputs=model)


# ---------------------------------------------------------------------------
# Multi-head routing helper
# ---------------------------------------------------------------------------


def _choose_head(
    out: tf.Tensor, num_heads: int, one_hot_task_id: tf.Tensor
) -> tf.Tensor:
    """Select the head corresponding to the current task.

    Args:
        out: Output tensor of shape ``(batch, n_actions * num_heads)``.
        num_heads: Number of output heads.
        one_hot_task_id: One-hot task vector of shape ``(batch, num_heads)``.

    Returns:
        Tensor of shape ``(batch, n_actions)`` for the active head.
    """
    batch_size = tf.shape(out)[0]
    out = tf.reshape(out, [batch_size, -1, num_heads])
    return tf.squeeze(out @ tf.expand_dims(one_hot_task_id, 2), axis=2)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


class MlpActor(_keras.Model):
    """Stochastic discrete-action policy.

    Returns unnormalised logits over the action space.  The SAC update uses
    a ``Categorical`` distribution built from these logits.

    Args:
        state_space: Observation space (``gymnasium.spaces.Box``).
        action_space: Action space (``gymnasium.spaces.Discrete``).
        num_tasks: Number of tasks (one-hot dimension).
        hidden_sizes: Dense-layer widths after the CNN.
        activation: Hidden-layer activation function.
        use_layer_norm: Apply layer normalisation after the first dense layer.
        use_lstm: Use LSTM pooling in the CNN head.
        num_heads: Number of output heads (1 = shared head, >1 = multi-head).
        hide_task_id: Do not concatenate the task one-hot to CNN features.
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
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hide_task_id = hide_task_id

        self.core = mlp(
            state_space.shape,
            num_tasks,
            hidden_sizes,
            activation,
            use_layer_norm,
            use_lstm,
            hide_task_id,
        )
        self.head_mu = Sequential(
            [Input(shape=(hidden_sizes[-1],)), Dense(action_space.n * num_heads)]
        )
        self.action_space = action_space

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        """Forward pass: return action logits.

        Args:
            obs: Batch of observations, shape ``(batch, H, W, C)``.
            one_hot_task_id: Batch of task vectors, shape ``(batch, num_tasks)``.

        Returns:
            Logits tensor of shape ``(batch, n_actions)``.
        """
        features = (
            self.core(obs) if self.hide_task_id else self.core([obs, one_hot_task_id])
        )
        mu = self.head_mu(features)
        if self.num_heads > 1:
            mu = _choose_head(mu, self.num_heads, one_hot_task_id)
        return mu

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Trainable variables shared across all tasks (excludes per-task heads)."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        return self.core.trainable_variables + self.head_mu.trainable_variables


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class MlpCritic(_keras.Model):
    """Q-value network for discrete actions.

    Returns Q(s, a) for all actions simultaneously as a vector of length
    ``n_actions``.  The SAC update takes the expected Q-value using the
    current policy probabilities.

    Args:
        state_space: Observation space.
        action_space: Action space.
        num_tasks: Number of tasks (one-hot dimension).
        hidden_sizes: Dense-layer widths after the CNN.
        activation: Hidden-layer activation function.
        use_layer_norm: Apply layer normalisation after the first dense layer.
        use_lstm: Use LSTM pooling in the CNN head.
        num_heads: Number of output heads.
        hide_task_id: Do not concatenate the task one-hot to CNN features.
    """

    def __init__(
        self,
        state_space: gymnasium.spaces.Box,
        action_space: gymnasium.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super().__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = num_heads

        self.core = mlp(
            state_space.shape,
            num_tasks,
            hidden_sizes,
            activation,
            use_layer_norm,
            use_lstm,
            hide_task_id,
        )
        self.head = Sequential(
            [Input(shape=(hidden_sizes[-1],)), Dense(num_heads * action_space.n)]
        )

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        """Forward pass: return Q-values for all actions.

        Args:
            obs: Batch of observations, shape ``(batch, H, W, C)``.
            one_hot_task_id: Batch of task vectors, shape ``(batch, num_tasks)``.

        Returns:
            Q-value tensor of shape ``(batch, n_actions)``.
        """
        features = (
            self.core(obs) if self.hide_task_id else self.core([obs, one_hot_task_id])
        )
        value = self.head(features)
        if self.num_heads > 1:
            value = _choose_head(value, self.num_heads, one_hot_task_id)
        return value

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Trainable variables shared across all tasks (excludes per-task heads)."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        return self.core.trainable_variables + self.head.trainable_variables
