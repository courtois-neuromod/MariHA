mariha.methods
==============

Nine continual learning baseline implementations, all built on the
:class:`~mariha.methods.base.CLMethod` ABC.  CL methods are *composed*
onto agents after construction (``agent.cl_method = ...``) rather than
inheriting from a specific RL algorithm — every method works on
:class:`~mariha.rl.sac.SAC`, :class:`~mariha.rl.ppo.PPO`, and
:class:`~mariha.rl.dqn.DQN` through the agent-agnostic hooks
:meth:`~mariha.rl.base.BaseAgent.get_named_parameter_groups`,
:meth:`~mariha.rl.base.BaseAgent.forward_for_importance`, and
:meth:`~mariha.rl.base.BaseAgent.distill_targets`.

.. seealso:: :doc:`/cl_methods` — grouped conceptual overview with math and references.

mariha.methods.base
-------------------

The :class:`~mariha.methods.base.CLMethod` ABC defining the hook surface
that every concrete method overrides.

.. automodule:: mariha.methods.base
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.regularizer_base
-------------------------------

Shared bookkeeping for parameter-regularization methods (L2, EWC, MAS,
SI).  Maintains the ``θ*`` snapshots and per-parameter importance
weights so concrete subclasses only have to implement
``_compute_importance``.

.. automodule:: mariha.methods.regularizer_base
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.distillation_base
--------------------------------

Shared episodic-memory + distillation gradient injection for DER and
ClonEx.  Dispatches to softmax KL on actor logits (SAC/PPO), soft KL
with temperature on Q-values (DQN — Rusu et al. 2016), and MSE on
critic / value heads (ClonEx).

.. automodule:: mariha.methods.distillation_base
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.l2
-----------------

.. automodule:: mariha.methods.l2
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.ewc
------------------

.. automodule:: mariha.methods.ewc
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.mas
------------------

.. automodule:: mariha.methods.mas
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.si
-----------------

.. automodule:: mariha.methods.si
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.agem
-------------------

.. automodule:: mariha.methods.agem
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.packnet
-----------------------

.. automodule:: mariha.methods.packnet
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.der
------------------

.. automodule:: mariha.methods.der
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.clonex
---------------------

.. automodule:: mariha.methods.clonex
   :members:
   :undoc-members:
   :show-inheritance:

mariha.methods.multitask
------------------------

.. automodule:: mariha.methods.multitask
   :members:
   :undoc-members:
   :show-inheritance:
