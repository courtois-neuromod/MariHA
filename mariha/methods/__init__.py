"""Continual learning methods for MariHA.

Available methods
-----------------
- ``sac``       — Vanilla SAC (no CL).
- ``l2``        — L2 regularisation (uniform importance).
- ``ewc``       — Elastic Weight Consolidation (Fisher diagonal).
- ``mas``       — Memory Aware Synapses (output sensitivity).
- ``si``        — Synaptic Intelligence (online surrogate).
- ``owl``       — EWC + UCB1 bandit task weighting.
- ``packnet``   — Iterative network pruning.
- ``agem``      — Averaged Gradient Episodic Memory.
- ``vcl``       — Variational Continual Learning.
- ``der``       — Dark Experience Replay++ (actor distillation).
- ``clonex``    — ClonEx (actor + critic distillation).
- ``multitask`` — Joint training (shared replay, upper bound).
"""

from mariha.methods.agem import AGEM_SAC
from mariha.methods.clonex import ClonEx_SAC
from mariha.methods.der import DER_SAC
from mariha.methods.ewc import EWC_SAC
from mariha.methods.l2 import L2_SAC
from mariha.methods.mas import MAS_SAC
from mariha.methods.multitask import MultiTask_SAC
from mariha.methods.owl import OWL_SAC
from mariha.methods.packnet import PackNet_SAC
from mariha.methods.regularization import Regularization_SAC
from mariha.methods.si import SI_SAC
from mariha.methods.vcl import VCL_SAC

__all__ = [
    "AGEM_SAC",
    "ClonEx_SAC",
    "DER_SAC",
    "EWC_SAC",
    "L2_SAC",
    "MAS_SAC",
    "MultiTask_SAC",
    "OWL_SAC",
    "PackNet_SAC",
    "Regularization_SAC",
    "SI_SAC",
    "VCL_SAC",
]
