"""Continual learning methods for MariHA.

Composition-based CL methods.  Each :class:`CLMethod` subclass is
attached to an agent after construction via ``agent.cl_method = ...``.
The concrete methods all register themselves with
:func:`mariha.benchmark.cl_registry.register_cl` at import time, so
importing this package is sufficient to populate the registry used by
``--cl_method`` on the CLI.

Importing the concrete subclasses (rather than only the base classes)
is what triggers the ``@register_cl`` decorators, so the order matters
less than the fact that *every* method module is imported here.
"""

from mariha.methods.base import CLMethod
from mariha.methods.distillation_base import DistillationMethod
from mariha.methods.regularizer_base import ParameterRegularizer

# Importing the concrete subclasses triggers their @register_cl
# decorators.  Order alphabetical for readability — registry order
# does not matter.
from mariha.methods.agem import AGEM
from mariha.methods.clonex import ClonEx
from mariha.methods.der import DER
from mariha.methods.ewc import EWC
from mariha.methods.l2 import L2Regularizer
from mariha.methods.mas import MAS
from mariha.methods.multitask import MultiTask
from mariha.methods.packnet import PackNet
from mariha.methods.si import SI

__all__ = [
    "CLMethod",
    "DistillationMethod",
    "ParameterRegularizer",
    "AGEM",
    "ClonEx",
    "DER",
    "EWC",
    "L2Regularizer",
    "MAS",
    "MultiTask",
    "PackNet",
    "SI",
]
