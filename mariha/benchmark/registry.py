"""Algorithm registry for the MariHA benchmark.

All built-in algorithms are registered in ``mariha/rl/__init__.py``.
New algorithms register themselves with the ``@register`` decorator or by
calling ``register(name)(cls)`` directly.

Usage::

    from mariha.benchmark.registry import register, get_agent_class

    @register("my_algo")
    class MyAlgo(BenchmarkAgent):
        ...

    # Later:
    agent_cls = get_agent_class("my_algo")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mariha.benchmark.agent import BenchmarkAgent

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator that registers an algorithm class under ``name``.

    Args:
        name: Case-insensitive string key used with ``--algorithm``.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Example::

        @register("ddqn")
        class DDQN(BenchmarkAgent):
            ...
    """
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_agent_class(name: str) -> type:
    """Look up an algorithm class by name.

    Args:
        name: Algorithm name (case-insensitive), e.g. ``"sac"``, ``"ewc"``.

    Returns:
        The registered ``BenchmarkAgent`` subclass.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available algorithms: {available}"
        )
    return _REGISTRY[key]


def list_agents() -> list[str]:
    """Return a sorted list of all registered algorithm names."""
    return sorted(_REGISTRY)
