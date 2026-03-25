Continual Learning Methods
==========================

MariHA provides 12 CL baseline implementations, all of which extend
:class:`~mariha.rl.sac.SAC` through its five hook methods.  The methods
are grouped below by mechanism.

Selecting a method
------------------

Pass ``--cl_method <name>`` to ``mariha-run-cl``:

.. code-block:: bash

   mariha-run-cl --subject sub-01 --cl_method ewc --seed 0

Use ``None`` (default, omit the flag) for vanilla SAC — sequential
fine-tuning with no forgetting prevention.

.. _regularisation-methods:

Regularisation methods
----------------------

These methods add a penalty term to the loss that discourages deviation
from the parameter values learned on previous tasks.  They share the
:class:`~mariha.methods.regularization.Regularization_SAC` base class,
which maintains a **snapshot of old parameters** and a **per-parameter
importance weight** vector.

The penalty is:

.. math::

   \mathcal{L}_{\text{reg}} = \lambda \sum_k \Omega_k \cdot (\theta_k - \theta_k^*)^2

where :math:`\theta_k^*` is the parameter snapshot from the end of the
previous task and :math:`\Omega_k` is the importance weight.

``l2`` — L2 Regularisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.l2.L2_SAC` uses **uniform importance weights**
(:math:`\Omega_k = 1`).  All parameters are penalised equally regardless
of how critical they are to past tasks.  This is the simplest baseline
and often used as a lower bound for regularisation methods.

``ewc`` — Elastic Weight Consolidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.ewc.EWC_SAC` approximates importance with the
**diagonal of the Fisher information matrix**, computed from Jacobians of
the policy logits and Q-values w.r.t. parameters:

.. math::

   F_k = \mathbb{E}\left[\left(\frac{\partial \log \pi}{\partial \theta_k}\right)^2\right]

Reference: Kirkpatrick et al., 2017 — `arXiv:1612.00796 <https://arxiv.org/abs/1612.00796>`_.

``mas`` — Memory Aware Synapses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.mas.MAS_SAC` measures importance as the
**sensitivity of the output magnitude** to parameter perturbations:

.. math::

   \Omega_k = \mathbb{E}\left[\left|\frac{\partial \|\mathbf{f}(\mathbf{x})\|^2}{\partial \theta_k}\right|\right]

This is task-agnostic — no labels are needed, only the unlabelled input
distribution from the replay buffer.

Reference: Aljundi et al., 2018 — `arXiv:1711.09601 <https://arxiv.org/abs/1711.09601>`_.

``si`` — Synaptic Intelligence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.si.SI_SAC` accumulates importance **online during
training**, at each gradient step:

.. math::

   \omega_k \mathrel{+}= -g_k \cdot (\theta_k - \theta_k^*)

At the task boundary it consolidates into:

.. math::

   \Omega_k \mathrel{+}= \max\!\left(\frac{\omega_k}{(\theta_k - \theta_k^*)^2 + \xi}, 0\right)

This avoids the expensive Jacobian computation needed by EWC and MAS.

Reference: Zenke et al., 2017 — `arXiv:1703.04200 <https://arxiv.org/abs/1703.04200>`_.

``owl`` — Online continual learning Without confLict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.owl.OWL_SAC` combines three mechanisms to prevent
catastrophic forgetting without requiring task identity at test time:

1. **Multi-head architecture** — one task-specific output head per task.
   The shared trunk is conditioned only on pixel observations
   (``hide_task_id=True``); the task one-hot is used only to route to the
   correct head after the trunk.  Learned features are therefore
   task-agnostic.

2. **EWC on the shared trunk** — Fisher-diagonal importance weights are
   computed only on the trunk parameters (automatically enforced: when
   ``num_heads > 1``, ``common_variables`` excludes the per-task heads).

3. **EWAF bandit** (Exponentially Weighted Average Forecaster) — tracks a
   probability distribution over task heads based on TD-error feedback.
   At test time when task identity is unknown, the head with the highest
   probability is used via
   :meth:`~mariha.methods.owl.OWL_SAC.get_bandit_action`.

The EWAF update rule (importance-weighted exponential weights):

.. math::

   \log w_i \;\mathrel{-}=\; \frac{\eta \cdot \ell_t}{p_t(i)} \;\cdot\; \mathbf{1}[i = i_t]

where :math:`\ell_t` is the mean TD error on task :math:`i_t` and
:math:`p_t(i) = \operatorname{softmax}(\log \mathbf{w})_i`.

Reference: Kessler et al., 2022 — `arXiv:2106.02940 <https://arxiv.org/abs/2106.02940>`_.

.. _gradient-projection:

Gradient projection
-------------------

``agem`` — Averaged Gradient Episodic Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.agem.AGEM_SAC` stores a fixed-size
**episodic memory** of transitions from past tasks.  At each gradient
step, the current gradient :math:`\mathbf{g}` is projected onto the
feasible half-space defined by the reference gradient
:math:`\mathbf{g}_{\text{ref}}` (computed from the episodic memory):

.. math::

   \tilde{\mathbf{g}} = \begin{cases}
       \mathbf{g} & \text{if } \mathbf{g} \cdot \mathbf{g}_{\text{ref}} \geq 0 \\
       \mathbf{g} - \dfrac{\mathbf{g} \cdot \mathbf{g}_{\text{ref}}}{\|\mathbf{g}_{\text{ref}}\|^2}\, \mathbf{g}_{\text{ref}} & \text{otherwise}
   \end{cases}

The episodic memory is filled at the start of each new task by sampling
``episodic_mem_per_task`` transitions from the current replay buffer.

Reference: Chaudhry et al., 2019 — `arXiv:1812.00420 <https://arxiv.org/abs/1812.00420>`_.

.. _structural-methods:

Structural methods
------------------

``packnet`` — PackNet
~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.packnet.PackNet_SAC` uses **iterative weight
pruning** to allocate disjoint subnetworks to each task.  At the end of
each task, the bottom ``prune_perc`` fraction of currently-active actor
weights (by magnitude) are frozen and protected from future updates.

After each optimizer step, Adam-drifted frozen weights are restored from
their stored values to ensure they never change — a correctness step that
gradient masking alone would miss.

Unlike COOM's dynamic ``num_tasks_left / (num_tasks_left + 1)`` schedule,
MariHA uses a fixed ``prune_perc`` (default 0.5) because the total task
count (313+) is not known during scheduling.

Reference: Mallya & Lazebnik, 2018 — `arXiv:1711.05769 <https://arxiv.org/abs/1711.05769>`_.

.. _bayesian-methods:

Bayesian methods
----------------

``vcl`` — Variational Continual Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.vcl.VCL_SAC` replaces the standard dense layers
in the actor trunk with **Bayesian dense layers** whose weights follow
distributions :math:`\mathcal{N}(\mu, \sigma^2)` instead of point
estimates.

During training, weights are sampled via the reparameterisation trick.
The regularisation loss is the **KL divergence** between the current
posterior and the prior set at the previous task boundary:

.. math::

   \mathcal{L}_{\text{VCL}} = \lambda \cdot \text{KL}\!\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(\mu^*, \sigma^{*2})\right)

At each task start, ``update_prior()`` is called on all Bayesian layers,
setting :math:`(\mu^*, \sigma^{*2}) \leftarrow (\mu, \sigma^2)`.

The :class:`~mariha.methods.vcl.BayesianDense` layer is a proper Keras
``Layer`` subclass and is discovered automatically by traversing
``actor.core.layers``.

Reference: Nguyen et al., 2018 — `arXiv:1710.10628 <https://arxiv.org/abs/1710.10628>`_.

.. _experience-replay-methods:

Experience replay / distillation methods
-----------------------------------------

``der`` — Dark Experience Replay++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.der.DER_SAC` stores ``(obs, one_hot,
actor_logits)`` snapshots in an :class:`~mariha.replay.buffers.EpisodicMemory`
at each task boundary.  During subsequent tasks, a **KL distillation loss**
anchors the current actor to the stored logits:

.. math::

   \mathcal{L}_{\text{KL}} = \alpha \cdot \mathbb{E}\left[\text{KL}\!\left(\pi_{\text{stored}} \,\|\, \pi_{\text{current}}\right)\right]

The KL gradients are computed with a separate ``GradientTape`` inside
``adjust_gradients()`` and added directly to the actor gradients.

Reference: Buzzega et al., 2020 — `arXiv:2004.07211 <https://arxiv.org/abs/2004.07211>`_.

``clonex`` — ClonEx-SAC
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.clonex.ClonEx_SAC` extends DER++ by also
distilling the **critic Q-values** with an MSE loss:

.. math::

   \mathcal{L}_{\text{MSE}} = \alpha_c \cdot \mathbb{E}\left[\|Q_{\text{current}} - Q_{\text{stored}}\|^2\right]

Both actor (KL) and critic (MSE) distillation gradients are computed from
nested tapes and added to the respective gradient lists in
``adjust_gradients()``.

Reference: Ben-Iwhiwhu et al., 2021 — `arXiv:2105.07748 <https://arxiv.org/abs/2105.07748>`_.

.. _upper-lower-bounds:

Upper / lower bounds
--------------------

``multitask`` — Joint Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.methods.multitask.MultiTask_SAC` is vanilla SAC with
``reset_buffer_on_task_change=False``.  The replay buffer accumulates
data from *all* tasks throughout training, simulating joint training.
This is the **upper bound** — it ignores the sequential constraint and
has access to data from all past tasks at every gradient step.

*(none)* — Sequential Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vanilla :class:`~mariha.rl.sac.SAC` with default settings
(``reset_buffer_on_task_change=True``) fine-tunes sequentially on each
task with no forgetting prevention.  This is the **lower bound**.

Adding a new method
-------------------

Subclass :class:`~mariha.rl.sac.SAC` and override any of the five hooks:

.. code-block:: python

   from mariha.rl.sac import SAC

   class MyMethod(SAC):

       def on_task_start(self, current_task_idx: int) -> None:
           super().on_task_start(current_task_idx)
           # Snapshot parameters, fill episodic memory, etc.

       def get_auxiliary_loss(self, seq_idx) -> tf.Tensor:
           # Must use TF ops only (called inside @tf.function).
           loss = ...
           return tf.cond(seq_idx > 0, lambda: self.coef * loss, lambda: 0.0)

       def adjust_gradients(self, actor_grads, critic_grads, alpha_grad,
                             current_task_idx, metrics, episodic_batch=None):
           # Project or augment gradients.
           return actor_grads, critic_grads, alpha_grad

Then register it in ``scripts/run_cl.py``'s ``_build_cl_agent()`` dispatch.
