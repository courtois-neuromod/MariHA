"""Neural network architectures for MariHA RL agents.

All agents share a common convolutional backbone (:class:`BaseCNN`) ported
from the ``ppo_study`` reference:

    4× Conv2D(32, 3×3, stride=2, padding='same', ReLU, orthogonal init)
    → Flatten → Dense(512, ReLU, orthogonal init).

Algorithm-specific components sit on top of this shared trunk:

**SAC**
    :class:`MlpActor` (action logits) and :class:`MlpCritic` (Q-values for
    all actions) share :class:`BaseCNN`, concatenate the task one-hot to
    its 512-d output, optionally apply additional dense layers, then
    produce per-action outputs.

**PPO**
    :class:`PPOActorCritic` wraps :class:`BaseCNN` with separate actor
    (logits) and critic (scalar value) heads; task one-hot concatenated
    after the backbone.

**DQN**
    Reuses :class:`MlpCritic` as its Q-network.
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
# Shared CNN backbone (used by all agents)
# ---------------------------------------------------------------------------


BACKBONE_DIM = 512


class BaseCNN(_keras.Model):
    """Shared convolutional backbone for all MariHA agents.

    Architecture (from ``ppo_study/src/models.py::BaseModel``):

        4× Conv2D(32, 3×3, stride=2, padding='same', ReLU, orthogonal)
        → Flatten → Dense(512, ReLU, orthogonal).

    With ``use_lstm=True`` the convs are wrapped in ``TimeDistributed`` and
    the ``Flatten → Dense(512)`` projection is replaced by an
    ``LSTM(512, tanh)``.  The input is then expected to carry a leading
    time axis.

    Output shape: ``(batch, 512)``.

    Args:
        state_shape: Observation shape, e.g. ``(84, 84, 4)``.
        use_lstm: If ``True``, build the recurrent variant.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        use_lstm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        init = _keras.initializers.Orthogonal(gain=tf.sqrt(2.0))
        conv_layers = [
            Conv2D(
                32, 3, strides=2, padding="same", activation="relu",
                kernel_initializer=init, bias_initializer="zeros",
                name=f"conv{i + 1}",
            )
            for i in range(4)
        ]
        if use_lstm:
            self.convs = [TimeDistributed(c) for c in conv_layers]
            self.pool = TimeDistributed(Flatten())
            self.projection = LSTM(BACKBONE_DIM, activation="tanh", name="lstm_proj")
        else:
            self.convs = conv_layers
            self.pool = Flatten()
            self.projection = Dense(
                BACKBONE_DIM, activation="relu",
                kernel_initializer=init, bias_initializer="zeros",
                name="dense_proj",
            )
        self.output_dim = BACKBONE_DIM

    def call(self, obs: tf.Tensor) -> tf.Tensor:
        x = obs
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        return self.projection(x)


# ---------------------------------------------------------------------------
# Shared CNN + optional dense trunk (functional builder for SAC / DQN)
# ---------------------------------------------------------------------------


def mlp(
    state_shape: Tuple[int, ...],
    num_tasks: int,
    hidden_sizes: Iterable[int],
    activation: Callable,
    use_layer_norm: bool = False,
    use_lstm: bool = False,
    hide_task_id: bool = False,
) -> Model:
    """Build ``BaseCNN`` + optional post-trunk MLP, accepting ``(obs, task_id)``.

    The backbone produces a 512-d feature vector.  The task one-hot is
    concatenated to it (unless ``hide_task_id``), then any additional
    dense layers from ``hidden_sizes`` are applied.  With
    ``hidden_sizes=()`` the trunk output is returned directly.

    Args:
        state_shape: Observation shape, e.g. ``(84, 84, 4)``.
        num_tasks: Length of the one-hot task vector.
        hidden_sizes: Extra dense-layer widths applied *after* the shared
            512 trunk.  Default is ``()`` — no extra layers.
        activation: Activation for the extra dense layers.
        use_layer_norm: If ``True``, apply ``LayerNormalization`` + ``tanh``
            after the first extra dense layer.
        use_lstm: Forward to :class:`BaseCNN`.
        hide_task_id: If ``True``, the task one-hot is not concatenated.

    Returns:
        A Keras ``Model`` with inputs ``[obs, task_id]`` (or just ``obs``
        when ``hide_task_id=True``).
    """
    task_input = Input(shape=(num_tasks,), name="task_input", dtype=tf.float32)
    conv_in = Input(shape=state_shape, name="conv_head_in")
    backbone = BaseCNN(state_shape, use_lstm=use_lstm)
    features = backbone(conv_in)

    out = features if hide_task_id else Concatenate()([features, task_input])

    hidden_sizes = tuple(hidden_sizes)
    if hidden_sizes:
        out = Dense(hidden_sizes[0])(out)
        if use_layer_norm:
            out = LayerNormalization()(out)
            out = Activation(tf.nn.tanh)(out)
        else:
            out = Activation(activation)(out)
        for size in hidden_sizes[1:]:
            out = Dense(size, activation=activation)(out)

    inputs = conv_in if hide_task_id else [conv_in, task_input]
    return Model(inputs=inputs, outputs=out)


def _trunk_output_dim(num_tasks: int, hidden_sizes: Tuple[int, ...], hide_task_id: bool) -> int:
    """Width of the tensor feeding the per-algorithm heads."""
    if hidden_sizes:
        return hidden_sizes[-1]
    return BACKBONE_DIM if hide_task_id else BACKBONE_DIM + num_tasks


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
        hidden_sizes: Extra dense-layer widths after the shared 512 trunk.
            Default ``()`` — strict alignment with PPO.
        activation: Activation for the extra dense layers.
        use_layer_norm: Apply layer normalisation after the first extra
            dense layer.
        use_lstm: Use LSTM pooling in the shared backbone.
        num_heads: Number of output heads (1 = shared head, >1 = multi-head).
        hide_task_id: Do not concatenate the task one-hot to backbone features.
    """

    def __init__(
        self,
        state_space: gymnasium.spaces.Box,
        action_space: gymnasium.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Tuple[int, ...] = (),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hide_task_id = hide_task_id

        hidden_sizes = tuple(hidden_sizes)
        self.core = mlp(
            state_space.shape,
            num_tasks,
            hidden_sizes,
            activation,
            use_layer_norm,
            use_lstm,
            hide_task_id,
        )
        head_in = _trunk_output_dim(num_tasks, hidden_sizes, hide_task_id)
        self.head_mu = Sequential(
            [Input(shape=(head_in,)), Dense(action_space.n * num_heads)]
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
    ``n_actions``.  Used as the SAC critic and the DQN Q-network.

    Args:
        state_space: Observation space.
        action_space: Action space.
        num_tasks: Number of tasks (one-hot dimension).
        hidden_sizes: Extra dense-layer widths after the shared 512 trunk.
            Default ``()`` — strict alignment with PPO.
        activation: Activation for the extra dense layers.
        use_layer_norm: Apply layer normalisation after the first extra
            dense layer.
        use_lstm: Use LSTM pooling in the shared backbone.
        num_heads: Number of output heads.
        hide_task_id: Do not concatenate the task one-hot to backbone features.
    """

    def __init__(
        self,
        state_space: gymnasium.spaces.Box,
        action_space: gymnasium.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Iterable[int] = (),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super().__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = num_heads

        hidden_sizes = tuple(hidden_sizes)
        self.core = mlp(
            state_space.shape,
            num_tasks,
            hidden_sizes,
            activation,
            use_layer_norm,
            use_lstm,
            hide_task_id,
        )
        head_in = _trunk_output_dim(num_tasks, hidden_sizes, hide_task_id)
        self.head = Sequential(
            [Input(shape=(head_in,)), Dense(num_heads * action_space.n)]
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


# ---------------------------------------------------------------------------
# PPO actor-critic
# ---------------------------------------------------------------------------


class PPOActorCritic(_keras.Model):
    """Shared-backbone actor-critic for PPO.

    Wraps :class:`BaseCNN` with separate actor (logits) and critic
    (scalar value) heads.  Optionally concatenates a one-hot task vector
    to the backbone output before the heads.

    Args:
        state_space: Observation space.
        action_space: Discrete action space.
        num_tasks: Length of the one-hot task vector.
        hide_task_id: If ``True``, don't use the task one-hot.
    """

    def __init__(
        self,
        state_space: "gymnasium.spaces.Box",
        action_space: "gymnasium.spaces.Discrete",
        num_tasks: int,
        hide_task_id: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hide_task_id = hide_task_id
        self.num_tasks = num_tasks
        self.backbone = BaseCNN(state_space.shape)
        init = _keras.initializers.Orthogonal(gain=1.0)
        head_in = BACKBONE_DIM if hide_task_id else BACKBONE_DIM + num_tasks
        self.actor_head = Dense(action_space.n, kernel_initializer=init,
                                bias_initializer="zeros", name="actor_logits")
        self.critic_head = Dense(1, kernel_initializer=init,
                                 bias_initializer="zeros", name="critic_value")
        # Build heads eagerly so weights exist before first call.
        self.actor_head.build((None, head_in))
        self.critic_head.build((None, head_in))

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor = None):
        """Forward pass.

        Args:
            obs: Observation batch ``(batch, H, W, C)``.
            one_hot_task_id: Task one-hot ``(batch, num_tasks)``.

        Returns:
            Tuple ``(logits, value)`` where logits has shape
            ``(batch, n_actions)`` and value has shape ``(batch, 1)``.
        """
        features = self.backbone(obs)
        if not self.hide_task_id and one_hot_task_id is not None:
            features = tf.concat([features, one_hot_task_id], axis=-1)
        return self.actor_head(features), self.critic_head(features)
