"""Agent registry for the MariHA benchmark.

All built-in agents are registered in ``mariha/rl/__init__.py``.
New agents register themselves with the ``@register`` decorator or by
calling ``register(name)(cls)`` directly.

Usage::

    from mariha.benchmark.registry import register, get_agent_class

    @register("my_agent")
    class MyAgent(BenchmarkAgent):
        ...

    # Later:
    agent_cls = get_agent_class("my_agent")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mariha.benchmark.agent import BenchmarkAgent

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator that registers an agent class under ``name``.

    Args:
        name: Case-insensitive string key used with ``--agent``.

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
    """Look up an agent class by name.

    Args:
        name: Agent name (case-insensitive), e.g. ``"sac"``, ``"ewc"``.

    Returns:
        The registered ``BenchmarkAgent`` subclass.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"Unknown agent '{name}'. "
            f"Available agents: {available}"
        )
    return _REGISTRY[key]


def list_agents() -> list[str]:
    """Return a sorted list of all registered agent names."""
    return sorted(_REGISTRY)
