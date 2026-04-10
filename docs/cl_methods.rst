Continual Learning Methods
==========================

MariHA provides nine CL baseline implementations.  Every method is
implemented as a :class:`~mariha.methods.base.CLMethod` subclass that is
*composed* onto an agent after construction (``agent.cl_method = ...``)
rather than inheriting from a particular RL algorithm.  As a result the
same nine methods all work on every built-in agent
(:class:`~mariha.rl.sac.SAC`, :class:`~mariha.rl.ppo.PPO`,
:class:`~mariha.rl.dqn.DQN`) — and on any future agent that implements
the agent-agnostic CL hooks.

Selecting a method
------------------

Pass ``--agent`` to pick the RL algorithm and ``--cl_method`` to pick
the continual learning strategy that is composed on top of it:

.. code-block:: bash

   mariha-run-cl --agent sac --cl_method ewc --subject sub-01 --seed 0
   mariha-run-cl --agent ppo --cl_method packnet --subject sub-01 --seed 0
   mariha-run-cl --agent dqn --cl_method der --subject sub-01 --seed 0

Omit ``--cl_method`` to fine-tune sequentially with no forgetting
prevention — the standard lower bound for continual-learning
comparisons.

The agent-agnostic contract
---------------------------

Three :class:`~mariha.rl.base.BaseAgent` hooks are the only points where
a CL method touches an agent:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Hook
     - Returned by the agent
   * - ``get_named_parameter_groups()``
     - ``dict[str, list[tf.Variable]]`` — one entry per logical parameter
       group.  SAC returns ``{"actor", "critic1", "critic2"}``;
       PPO returns ``{"policy", "value"}``; DQN returns ``{"q"}``.
   * - ``forward_for_importance(obs, one_hot)``
     - ``dict[str, tf.Tensor]`` of differentiable per-group outputs.
       Used by EWC and MAS to compute Jacobian-based importance.
   * - ``distill_targets(obs, one_hot)``
     - ``dict[str, tf.Tensor]`` of distillation targets.  SAC/PPO
       return ``"actor_logits"``; DQN returns ``"q_values"``; ClonEx
       additionally consumes ``"critic1_q"``/``"critic2_q"`` (SAC) or
       ``"value"`` (PPO).

Methods that do not need data (L2, MultiTask) can ignore
``forward_for_importance`` and ``distill_targets``.

Default group resolution
~~~~~~~~~~~~~~~~~~~~~~~~

When a regularizer or pruning method is constructed without an explicit
``regularize_groups`` argument, it auto-resolves the target group by
trying ``"actor"`` → ``"policy"`` → ``"q"`` in that order and using the
first one present on the agent.  This gives the canonical
"regularize the actor / policy network" behaviour for SAC and PPO and
"regularize the Q-network" for DQN, with no per-agent configuration on
the user's part.

.. _regularisation-methods:

Regularisation methods
----------------------

These methods add a quadratic penalty to the loss that anchors the
current parameters to a snapshot ``θ*`` taken at the previous task
boundary.  They share the
:class:`~mariha.methods.regularizer_base.ParameterRegularizer` base
class, which maintains the snapshot and per-parameter importance
weights and adds the penalty inside the agent's gradient tape via
:meth:`~mariha.methods.base.CLMethod.compute_loss_penalty`.

The penalty is:

.. math::

   \mathcal{L}_{\text{reg}} = \lambda \sum_g \sum_k \Omega_{g,k} \cdot (\theta_{g,k} - \theta_{g,k}^*)^2

where :math:`g` ranges over the regularized parameter groups and
:math:`\Omega_{g,k}` is the importance weight of parameter
:math:`\theta_{g,k}`.

``l2`` — L2 Regularisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.l2.L2Regularizer` uses **uniform importance
weights** (:math:`\Omega_{g,k} = 1`).  All parameters in the regularized
group are penalised equally regardless of how critical they are to past
tasks.  This is the simplest baseline and the natural lower bound for
the regularisation family.  No data sampling is needed at task end.

``ewc`` — Elastic Weight Consolidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.ewc.EWC` approximates importance with the
**diagonal of the Fisher information matrix**, computed from Jacobians
of each per-group output (actor logits, Q-values, …) with respect to
the parameters:

.. math::

   F_k = \mathbb{E}\left[\left(\frac{\partial}{\partial \theta_k}\,\sum_o \text{output}_o\right)^2\right]

The expectation is averaged over a small batch of replay samples drawn
from the just-finished task at the moment :meth:`on_task_end` fires.
EWC was originally introduced on DQN where it reduces naturally to a
single ``"q"`` group; the same class also works for SAC's ``"actor"``
and PPO's ``"policy"`` group.

Reference: Kirkpatrick et al., 2017 — `arXiv:1612.00796 <https://arxiv.org/abs/1612.00796>`_.

``mas`` — Memory Aware Synapses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.mas.MAS` measures importance as the
**sensitivity of the squared output norm** to parameter perturbations:

.. math::

   \Omega_k = \mathbb{E}\left[\left|\frac{\partial \|\mathbf{f}(\mathbf{x})\|^2}{\partial \theta_k}\right|\right]

This is task-agnostic — no labels are needed, only the unlabelled input
distribution from the replay buffer.  Like EWC, MAS plugs into any
agent that implements
:meth:`~mariha.rl.base.BaseAgent.forward_for_importance`.

Reference: Aljundi et al., 2018 — `arXiv:1711.09601 <https://arxiv.org/abs/1711.09601>`_.

``si`` — Synaptic Intelligence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.si.SI` accumulates importance **online during
training** instead of computing it at the task boundary.  At every
gradient step the per-step surrogate is

.. math::

   \omega_k \mathrel{+}= -g_k \cdot (\theta_k - \theta_k^*)

(implemented in :meth:`~mariha.methods.base.CLMethod.after_gradient_step`,
which fires inside the agent's compiled update path).  At each task
boundary the surrogate is consolidated into the running importance
weight:

.. math::

   \Omega_k \mathrel{+}= \max\!\left(\frac{\omega_k}{(\theta_k - \theta_k^*)^2 + \xi}, 0\right)

This avoids the expensive Jacobian computation needed by EWC and MAS at
the cost of carrying a per-parameter accumulator throughout training.

Reference: Zenke et al., 2017 — `arXiv:1703.04200 <https://arxiv.org/abs/1703.04200>`_.

.. _gradient-projection:

Gradient projection
-------------------

``agem`` — Averaged Gradient Episodic Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.agem.AGEM` stores a fixed-size **episodic
memory** of transitions sampled at every task boundary.  At each
gradient step it computes a *reference gradient* from the episodic
memory via :meth:`~mariha.rl.base.BaseAgent.compute_reference_gradients`
(itself a template method that calls each agent's
:meth:`_compute_reference_loss`) and projects the live gradient onto
the half-space defined by it:

.. math::

   \tilde{\mathbf{g}} = \begin{cases}
       \mathbf{g} & \text{if } \mathbf{g} \cdot \mathbf{g}_{\text{ref}} \geq 0 \\
       \mathbf{g} - \dfrac{\mathbf{g} \cdot \mathbf{g}_{\text{ref}}}{\|\mathbf{g}_{\text{ref}}\|^2}\, \mathbf{g}_{\text{ref}} & \text{otherwise}
   \end{cases}

The projection is per-group rather than over the concatenated gradient,
matching the original SAC implementation; for DQN the per-group form
coincides with the original A-GEM formulation since there is only one
group.

Reference: Chaudhry et al., 2019 — `arXiv:1812.00420 <https://arxiv.org/abs/1812.00420>`_.

.. _structural-methods:

Structural methods
------------------

``packnet`` — PackNet
~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.packnet.PackNet` uses **iterative magnitude
pruning** to allocate disjoint subnetworks to each task.  At the end of
each task, the bottom ``prune_perc`` fraction of currently-active
weights (by magnitude) are frozen and protected from future updates.

Three mechanisms keep frozen weights stationary:

1. :meth:`~mariha.methods.base.CLMethod.adjust_gradients` zeroes out
   gradients of frozen parameters before the optimizer step.
2. :meth:`~mariha.methods.base.CLMethod.after_gradient_step` re-asserts
   the frozen value as ``var = var * mask + frozen``, undoing any
   Adam-momentum drift the optimizer might introduce on a zero
   gradient.
3. :meth:`~mariha.methods.base.CLMethod.on_task_end` selects the
   lowest-magnitude *free* weights and adds them to the frozen set.

Unlike COOM's dynamic ``num_tasks_left / (num_tasks_left + 1)``
schedule, MariHA uses a fixed ``prune_perc`` (default 0.5) because the
total task count (313+ scenes) is not known in advance.

Reference: Mallya & Lazebnik, 2018 — `arXiv:1711.05769 <https://arxiv.org/abs/1711.05769>`_.

.. _experience-replay-methods:

Experience replay / distillation methods
-----------------------------------------

These methods store snapshots of past-task targets and add a
distillation gradient to subsequent updates.  They share the
:class:`~mariha.methods.distillation_base.DistillationMethod` base,
which inspects the keys of ``agent.distill_targets()`` and dispatches
to the appropriate loss formula:

* ``"actor_logits"`` (SAC, PPO) → softmax KL
* ``"q_values"`` (DQN) → soft KL with temperature ``T`` (Rusu et al.
  2016, *Policy Distillation*, `arXiv:1511.06295 <https://arxiv.org/abs/1511.06295>`_)
* ``"critic1_q"`` / ``"critic2_q"`` (SAC, ClonEx only) → MSE
* ``"value"`` (PPO, ClonEx only) → MSE

The Q-value soft-KL uses ``T^2 · KL`` (the temperature-squared factor
preserves the gradient magnitude as ``T`` is varied).

``der`` — Dark Experience Replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.der.DER` stores ``(obs, one_hot, target)``
snapshots in a fixed-size episodic memory at every task boundary and
applies a KL distillation loss on subsequent gradient steps:

.. math::

   \mathcal{L}_{\text{KL}} = \alpha \cdot \mathbb{E}\left[\text{KL}\!\left(\pi_{\text{stored}} \,\|\, \pi_{\text{current}}\right)\right]

DER is the actor-only variant — only the policy / Q-value head receives
a distillation gradient.  The distillation gradient is computed in a
nested ``GradientTape`` inside :meth:`adjust_gradients` and added to
the actor / policy / Q gradient list.

Reference: Buzzega et al., 2020 — `arXiv:2004.07211 <https://arxiv.org/abs/2004.07211>`_.

``clonex`` — ClonEx
~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.clonex.ClonEx` extends DER by also distilling
the **critic / value head** with an MSE loss:

.. math::

   \mathcal{L}_{\text{MSE}} = \alpha_c \cdot \mathbb{E}\left[\|f_{\text{current}}(x) - f_{\text{stored}}(x)\|^2\right]

On SAC both critic1 and critic2 are distilled; on PPO the scalar value
head is distilled; on DQN the ``clone_critic`` flag is a no-op (the
single ``q_values`` target already provides the only available signal).

Reference: Ben-Iwhiwhu et al., 2021 — `arXiv:2105.07748 <https://arxiv.org/abs/2105.07748>`_.

.. _upper-lower-bounds:

Upper / lower bounds
--------------------

``multitask`` — Joint Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.multitask.MultiTask` is the joint-training
upper bound.  It overrides only :meth:`on_task_start` to set
``agent.reset_buffer_on_task_change = False`` so the replay buffer
accumulates data from *all* tasks throughout training.  The agent is
otherwise unmodified — every gradient step samples uniformly over the
union of every task seen so far, ignoring the sequential constraint.
This is the standard upper bound for CL benchmarks.

*(none)* — Sequential Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Omitting ``--cl_method`` runs the agent in vanilla mode with default
task-boundary resets (``reset_buffer_on_task_change=True``).  This
fine-tunes sequentially on each task with no forgetting prevention and
serves as the **lower bound** for any CL strategy.

Adding a new method
-------------------

See :doc:`adding_a_cl_method` for the full walk-through.  In short:

1. Subclass :class:`~mariha.methods.base.CLMethod` (or one of the two
   intermediate bases:
   :class:`~mariha.methods.regularizer_base.ParameterRegularizer` for
   anchor-style methods,
   :class:`~mariha.methods.distillation_base.DistillationMethod` for
   memory-replay distillation methods).
2. Override only the hooks you need (``on_task_end``,
   ``compute_loss_penalty``, ``adjust_gradients``,
   ``after_gradient_step``, ``get_episodic_batch``).
3. Decorate the class with
   :func:`~mariha.benchmark.cl_registry.register_cl` so it becomes
   selectable from the CLI as ``--cl_method <name>``.
4. Implement :meth:`~mariha.methods.base.CLMethod.add_args` and
   :meth:`~mariha.methods.base.CLMethod.from_args` for any
   hyperparameters.

The new class will work on every registered RL agent (SAC, PPO, DQN,
and any future addition) without any agent-side changes.
