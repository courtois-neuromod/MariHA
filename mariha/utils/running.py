"""Runtime utilities: seeding, weight/optimizer reset, type helpers.

Ported from COOM with VizDoom / Neptune / mrunner dependencies removed.
"""

from __future__ import annotations

import argparse
import random
import string
from datetime import datetime
from typing import Callable, Dict, Optional, Type, Union

import gymnasium
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Type-coercion helpers (used by argument parsers)
# ---------------------------------------------------------------------------


def str2bool(v: Union[bool, str]) -> bool:
    """Parse a boolean command-line argument."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def sci2int(v: str) -> int:
    """Convert scientific notation string (e.g. ``'1e5'``) to ``int``."""
    return int(float(v))


def float_or_str(v: Union[float, str]) -> Union[float, str]:
    """Return ``float(v)`` if possible, otherwise return ``v`` as-is."""
    try:
        return float(v)
    except ValueError:
        return v


def get_activation_from_str(name: str) -> Callable:
    """Return a TensorFlow activation function by name."""
    mapping = {
        "tanh": tf.tanh,
        "relu": tf.nn.relu,
        "elu": tf.nn.elu,
        "lrelu": tf.nn.leaky_relu,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(mapping)}")
    return mapping[name]


# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------


def set_seed(seed: int, env: Optional[gymnasium.Env] = None) -> None:
    """Set global random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value.
        env: If provided, also seeds ``env.action_space``.
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.action_space.seed(seed)


def get_readable_timestamp() -> str:
    """Return a human-readable timestamp string, e.g. ``'2026_03_01__12_00_00'``."""
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def get_random_string(n: int = 6) -> str:
    """Return a random alphanumeric string of length ``n``."""
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(n))


# ---------------------------------------------------------------------------
# Model / optimizer utilities
# ---------------------------------------------------------------------------


def reset_optimizer(optimizer: tf.keras.optimizers.Optimizer) -> None:
    """Reset all optimizer slot variables and the step counter to zero.

    Args:
        optimizer: A Keras optimizer instance.
    """
    if hasattr(optimizer, "iterations"):
        optimizer.iterations.assign(0)
    for var in optimizer.variables()[1:]:
        var.assign(tf.zeros_like(var))


def reset_weights(
    model: tf.keras.Model,
    model_cl: Type[tf.keras.Model],
    model_kwargs: Dict,
) -> None:
    """Re-initialise a model's weights by constructing a fresh instance.

    Args:
        model: The model whose weights will be overwritten.
        model_cl: The model class (must accept ``**model_kwargs``).
        model_kwargs: Keyword arguments passed to ``model_cl``.
    """
    dummy = model_cl(**model_kwargs)
    model.set_weights(dummy.get_weights())


# ---------------------------------------------------------------------------
# Task ID helpers
# ---------------------------------------------------------------------------


def create_one_hot_vec(num_tasks: int, task_id: int) -> np.ndarray:
    """Return a one-hot float32 vector of length ``num_tasks`` with index ``task_id`` set.

    Args:
        num_tasks: Total number of tasks.
        task_id: Index of the active task.

    Returns:
        ``np.ndarray`` of shape ``(num_tasks,)`` with dtype ``float32``.
    """
    vec = np.zeros(num_tasks, dtype=np.float32)
    vec[task_id] = 1.0
    return vec
