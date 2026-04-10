"""Continual learning method registry for MariHA.

Companion to :mod:`mariha.benchmark.registry` (which holds *agents*).
CL methods register themselves with the ``@register_cl`` decorator and
are selected at the command line via ``--cl_method <name>`` independently
of ``--agent <name>``.

This composition pattern decouples the algorithm from the CL strategy:
any registered CL method can be paired with any registered agent that
implements the agent-agnostic CL contract
(:meth:`~mariha.rl.base.BaseAgent.get_named_parameter_groups`,
:meth:`~mariha.rl.base.BaseAgent.forward_for_importance`,
:meth:`~mariha.rl.base.BaseAgent.distill_targets`).

Usage::

    from mariha.benchmark.cl_registry import register_cl, get_cl_class

    @register_cl("ewc")
    class EWC(ParameterRegularizer):
        ...

    cl_cls = get_cl_class("ewc")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mariha.methods.base import CLMethod  # noqa: F401

_REGISTRY: dict[str, type] = {}


def register_cl(name: str):
    """Decorator that registers a CL method class under ``name``.

    Args:
        name: Case-insensitive string key used with ``--cl_method``.

    Returns:
        A decorator that registers the class and returns it unchanged.
    """
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_cl_class(name: str) -> type:
    """Look up a CL method class by name.

    Args:
        name: CL method name (case-insensitive), e.g. ``"ewc"``, ``"l2"``.

    Returns:
        The registered :class:`~mariha.methods.base.CLMethod` subclass.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown CL method '{name}'. "
            f"Available CL methods: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_cl_methods() -> list[str]:
    """Return a sorted list of all registered CL method names."""
    return sorted(_REGISTRY)
