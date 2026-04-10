"""Built-in agent registry for MariHA.

Importing this module registers all built-in agents (SAC, PPO, DQN,
RandomAgent) in ``mariha.benchmark.registry``.  Continual-learning
methods are no longer registered as agents — they are composed onto
agents via ``agent.cl_method = ...``.  Use
``mariha.benchmark.cl_registry`` to look up CL classes by name.

Any script that needs to look up an agent by name should import this
module first::

    import mariha.rl  # registers all built-in agents
    from mariha.benchmark.registry import get_agent_class
    agent_cls = get_agent_class("ppo")
"""

from mariha.benchmark.registry import register

# Base SAC
from mariha.rl.sac import SAC
register("sac")(SAC)

# PPO
from mariha.rl.ppo import PPO
register("ppo")(PPO)

# DQN / DDQN
from mariha.rl.dqn import DQN
register("dqn")(DQN)
register("ddqn")(DQN)  # alias — double_dqn=True by default

# Random baseline
from mariha.rl.random.agent import RandomAgent
register("random")(RandomAgent)

__all__ = [
    "SAC",
    "PPO",
    "DQN",
    "RandomAgent",
]
