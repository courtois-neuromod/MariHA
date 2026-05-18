"""Declarative architecture specs for MariHA RL agents.

This module is the single, human-maintained description of each agent's
network topology. ``generate_architecture_diagram.py`` consumes it to render
the PNG panels under ``assets/model_architectures/``.

It is deliberately decoupled from ``mariha/rl/`` so the diagram tooling stays
lightweight (no TensorFlow import). Because the agent-level wiring — how many
networks exist, which are targets of which — cannot be reliably auto-derived
from code, it is declared here by hand. ``scripts/tests/test_architecture_specs.py``
introspects the real Keras models and fails if a spec drifts out of sync.

Every agent network shares the same ``BaseCNN`` backbone — 4 stride-2 conv
blocks -> Flatten -> Dense(512) -> Concatenate(task one-hot) — so a spec only
declares what *varies*: the set of networks, their output heads, and the
links drawn between networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Head:
    """An output head branching off a network's ``concat`` node."""

    id: str          # unique within its network
    label: str       # caption drawn on the box
    units: int       # output width (e.g. n_actions, or 1 for a scalar value)


@dataclass(frozen=True)
class Network:
    """One independent network: the shared backbone plus its output head(s)."""

    id: str                    # unique within the agent; also a node-name prefix
    title: str                 # human-readable label drawn beside the stack
    heads: Tuple[Head, ...]    # 1 head = straight output, 2 = branching heads


@dataclass(frozen=True)
class Link:
    """A weight-copy relationship between two networks.

    ``dst`` is a Polyak-tracked copy of ``src`` (a target network). It is
    rendered as a text note in the gap between the two stacks — not an
    arrow — because the target update is a *whole-network* weight copy
    (``theta_dst <- tau*theta_src + (1-tau)*theta_dst`` over every
    parameter), not a layer-to-layer connection.
    """

    src: str         # the network whose weights are copied (the online net)
    dst: str         # the network that receives the copy (the target net)


@dataclass(frozen=True)
class AgentArch:
    """An agent's full architecture: a panel of networks and their links."""

    agent: str
    title: str
    networks: Tuple[Network, ...]
    links: Tuple[Link, ...]


def build_specs(n_actions: int) -> Dict[str, AgentArch]:
    """Return the architecture registry, parameterised by action-space size.

    Keys are agent names. ``ddqn`` is intentionally absent — it shares DQN's
    architecture and is resolved to ``dqn`` by :func:`get_spec`.
    """
    q_head = Head("q", "Q-values", n_actions)

    # PPO: a single actor-critic network with a *shared* backbone feeding two
    # heads — action logits and a scalar state value.
    ppo = AgentArch(
        agent="ppo",
        title="PPO - Proximal Policy Optimization",
        networks=(
            Network("ac", "Actor-Critic", (
                Head("actor", "Actor (logits)", n_actions),
                Head("critic", "Critic (value)", 1),
            )),
        ),
        links=(),
    )

    # SAC: an actor plus twin Q-critics, each critic paired with a
    # Polyak-tracked target. Every network has its own independent backbone —
    # actor and critics share no weights. Only the Polyak weight-copy
    # relationships are drawn (each critic -> its target).
    sac = AgentArch(
        agent="sac",
        title="SAC - discrete Soft Actor-Critic",
        networks=(
            Network("actor", "Actor", (Head("logits", "Actor (logits)", n_actions),)),
            Network("critic1", "Critic 1", (q_head,)),
            Network("target_critic1", "Target Critic 1", (q_head,)),
            Network("critic2", "Critic 2", (q_head,)),
            Network("target_critic2", "Target Critic 2", (q_head,)),
        ),
        links=(
            Link("critic1", "target_critic1"),
            Link("critic2", "target_critic2"),
        ),
    )

    # DQN (and DDQN): an online Q-network and a Polyak-tracked target copy.
    dqn = AgentArch(
        agent="dqn",
        title="DQN - Deep Q-Network",
        networks=(
            Network("online", "Online Q-network", (q_head,)),
            Network("target", "Target Q-network", (q_head,)),
        ),
        links=(Link("online", "target"),),
    )

    return {"ppo": ppo, "sac": sac, "dqn": dqn}


def get_spec(model: str, n_actions: int) -> AgentArch:
    """Return the :class:`AgentArch` for ``model`` (``ddqn`` aliases ``dqn``)."""
    key = "dqn" if model == "ddqn" else model
    specs = build_specs(n_actions)
    if key not in specs:
        raise ValueError(
            f"No architecture spec for '{model}'. "
            f"Known: {sorted(specs)} (+ ddqn alias)."
        )
    return specs[key]
