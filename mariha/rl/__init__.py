"""Built-in agent registry for MariHA.

Importing this module registers all built-in agents (SAC, PPO, DQN,
SAC-based CL variants, and RandomAgent) in ``mariha.benchmark.registry``.

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

# SAC-based continual learning variants
from mariha.methods.l2 import L2_SAC
from mariha.methods.ewc import EWC_SAC
from mariha.methods.mas import MAS_SAC
from mariha.methods.si import SI_SAC
from mariha.methods.owl import OWL_SAC
from mariha.methods.packnet import PackNet_SAC
from mariha.methods.agem import AGEM_SAC
from mariha.methods.vcl import VCL_SAC
from mariha.methods.der import DER_SAC
from mariha.methods.clonex import ClonEx_SAC
from mariha.methods.multitask import MultiTask_SAC

register("l2")(L2_SAC)
register("ewc")(EWC_SAC)
register("mas")(MAS_SAC)
register("si")(SI_SAC)
register("owl")(OWL_SAC)
register("packnet")(PackNet_SAC)
register("agem")(AGEM_SAC)
register("vcl")(VCL_SAC)
register("der")(DER_SAC)
register("der++")(DER_SAC)   # alias
register("clonex")(ClonEx_SAC)
register("multitask")(MultiTask_SAC)

__all__ = [
    "SAC",
    "PPO",
    "DQN",
    "RandomAgent",
    "L2_SAC",
    "EWC_SAC",
    "MAS_SAC",
    "SI_SAC",
    "OWL_SAC",
    "PackNet_SAC",
    "AGEM_SAC",
    "VCL_SAC",
    "DER_SAC",
    "ClonEx_SAC",
    "MultiTask_SAC",
]
