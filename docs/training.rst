Training
========

MariHA's three built-in agents (:class:`~mariha.rl.sac.SAC`,
:class:`~mariha.rl.ppo.PPO`, :class:`~mariha.rl.dqn.DQN`) all subclass
:class:`~mariha.rl.base.BaseAgent` and share a single episode-driven
training loop owned by :class:`~mariha.rl.base.TrainingLoopRunner`.
Each agent implements only the algorithm-specific callbacks; the loop
itself, task-switch handling, session-boundary flushes, render
checkpoints, periodic logging and saves are all provided by the base
infrastructure.

Algorithms
----------

**Soft Actor-Critic (SAC)** — adapted for the discrete 9-action space:

* The actor outputs logits over the action space; a ``Categorical``
  distribution is built from them.
* Entropy is the Shannon entropy of the categorical distribution.
* Both critics output Q-values for *all* actions simultaneously
  (shape ``(batch, 9)``).
* The target entropy is ``0.98 × log(9) ≈ 2.09`` nats.
* Per-task ``log_alpha`` variables enable automatic entropy tuning.

**Proximal Policy Optimization (PPO)** — on-policy variant with
shared-trunk actor-critic, GAE advantages, clipped policy ratio, and
``n_epochs`` of mini-batch updates per rollout.  Sets
``update_granularity = "per_rollout"``: ``should_update`` returns True
only when the rollout buffer is full, so updates fire on a fixed-size
horizon rather than a per-step cadence.

**Deep Q-Network (DQN / Double DQN)** — value-based agent with
epsilon-greedy exploration, a target network refreshed via Polyak
averaging, and an optional Double-DQN target formulation
(``--double_dqn=True``, the default for the ``ddqn`` alias).

Model architecture
------------------

SAC's actor and critic both share the standard CNN + MLP trunk:

.. code-block:: text

   obs (84,84,4) ──► Conv(32, 8×8, stride 4)
                     Conv(64, 4×4, stride 2)
                     Conv(64, 3×3, stride 1)
                     Flatten  ──► (3136,)
                            │
   task_one_hot ────────────┤  Concatenate
                            │
                     Dense(256) → activation
                     Dense(256) → activation
                            │
                     head: Dense(9 × num_heads)

PPO uses a shared-trunk variant (:class:`~mariha.rl.models.PPOActorCritic`)
with separate policy and value heads after the dense trunk; DQN uses a
single Q-network with the same conv stack.

The CNN and dense trunk are the **common variables** — shared across
tasks and the natural target of regularisation-based CL methods.
Task-specific heads (when ``num_heads > 1``) are excluded from
regularisation by design.

Episode-driven training loop
-----------------------------

The runner runs until the curriculum is exhausted rather than a fixed
step budget.  Pseudocode for the per-step / off-policy path (SAC, DQN):

.. code-block:: python

   obs, info = env.reset()
   one_hot = info["task_one_hot"]
   agent.on_task_start(task_idx, scene_id)

   while True:
       action, extras = agent.select_action(
           obs=obs, one_hot=one_hot, global_step=step, ...
       )
       next_obs, r, terminated, truncated, info = env.step(action)
       agent.store_transition(
           obs=obs, action=action, reward=r, next_obs=next_obs,
           terminated=terminated, one_hot=one_hot, ..., extras=extras,
       )

       if agent.should_update(step):
           agent.update_step(global_step=step, current_task_idx=task_idx)

       if terminated or truncated:
           obs, info = env.reset()
           one_hot = info["task_one_hot"]
           if info["task_switch"]:
               agent.on_task_end(task_idx)         # CL hook fires here
               agent.on_task_change(new_idx, scene_id)  # buffer/opt resets
               agent.on_task_start(new_idx, scene_id)   # CL hook fires here
           if env.is_done:
               break
       else:
           obs = next_obs

PPO follows the same skeleton with ``update_granularity = "per_rollout"``:
``should_update`` only fires when the rollout buffer is full, and
``update_step`` runs ``n_epochs`` of minibatch passes followed by a
buffer reset.  Both modes share the same task-switch and
session-boundary handling because the runner is generic.

Task lifecycle
--------------

When the runner observes ``info["task_switch"] = True`` after a reset,
the boundary handling is:

1. ``agent.on_task_end(prev_task_idx)`` — the agent forwards this to
   ``self.cl_method.on_task_end(self, prev_task_idx)``.  The replay
   buffer still holds the just-finished task's data, so importance
   estimators (EWC/MAS/L2/SI), PackNet's pruning, and the distillation
   episodic-memory snapshot all run here.
2. ``agent.on_task_change(new_task_idx, new_scene_id)`` — agent-side
   resets: the replay buffer (``reset_buffer_on_task_change``),
   actor/critic weights (``reset_*_on_task_change``), and optimizer
   slot variables (``reset_optimizer_on_task_change``) are
   re-initialised if the corresponding flags are set.  The agent may
   also recompile its ``learn_on_batch`` ``tf.function`` so the new
   task index is baked into the trace.
3. ``agent.on_task_start(new_task_idx, new_scene_id)`` — the agent
   forwards this to ``self.cl_method.on_task_start(self,
   new_task_idx)``.  SI uses this hook to snapshot ``θ*`` and zero its
   small-omega accumulator.

CL composition points
----------------------

Continual learning methods are composed onto an agent after
construction (``agent.cl_method = SomeCLMethod(...)``) rather than
inherited.  Each agent invokes its attached
:class:`~mariha.methods.base.CLMethod` at five places:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Hook
     - Purpose
   * - ``on_task_start(agent, task_idx)``
     - Called once when a new task begins, after the agent has finished
       its task-boundary resets.  Eager Python — numpy operations
       safe.
   * - ``on_task_end(agent, task_idx)``
     - Called once when the just-finished task ends, **before** the
       agent resets state.  Importance estimators, pruning, and
       episodic-memory filling run here.
   * - ``compute_loss_penalty(agent, task_idx) → tf.Tensor``
     - Scalar regularization term added inside the agent's
       ``GradientTape``.  Used by L2/EWC/MAS/SI.  Default returns
       ``tf.zeros([])``.
   * - ``adjust_gradients(agent, grads_by_group, ...) → grads_by_group``
     - Modifies gradients before the optimizer step.  Used by AGEM
       (projection), PackNet (mask zeroing), DER/ClonEx (additive
       distillation gradients).  Default is identity.
   * - ``after_gradient_step(agent, grads_by_group, ...) → None``
     - Called after the optimizer applied the (adjusted) gradients.
       Runs inside the agent's ``@tf.function``-traced update path.
       SI uses this to accumulate its omega surrogate; PackNet uses
       it to re-assert frozen weight values.
   * - ``get_episodic_batch(agent, task_idx) → Optional[batch]``
     - Provides a replay batch from past tasks; the agent forwards it
       verbatim to ``adjust_gradients``.  Used by AGEM, DER, ClonEx.

The agent talks to the CL method through three additional contracts on
:class:`~mariha.rl.base.BaseAgent`:

* :meth:`~mariha.rl.base.BaseAgent.get_named_parameter_groups`
* :meth:`~mariha.rl.base.BaseAgent.forward_for_importance`
* :meth:`~mariha.rl.base.BaseAgent.distill_targets`

Together they let the same nine CL methods work on every RL agent
without any algorithm-specific code paths.  See :doc:`cl_methods` for
the full method catalogue and :doc:`adding_a_cl_method` for the
walk-through of writing a new one.

Replay buffers
--------------

The buffer type is set via the ``--buffer_type`` argument:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Type
     - Description
   * - ``fifo`` (default)
     - FIFO ring buffer.  Discards oldest transitions when full.
   * - ``reservoir``
     - Reservoir sampling — maintains a uniform distribution over all
       transitions seen, not just the most recent.
   * - ``priority``
     - Prioritised experience replay (SumTree implementation).
   * - ``per``
     - PER with SegmentTree and cleaner importance-sampling weights.

Off-policy agents (SAC, DQN) optionally use a per-scene buffer pool
(``--buffer_mode per_scene``), which keeps one buffer per scene and
flushes pending transitions on a session boundary.  PPO uses a
:class:`~mariha.replay.buffers.RolloutBuffer` instead of the off-policy
buffer family.

Single-scene training (``run_single.py``)
-----------------------------------------

``mariha-run-single`` trains any registered agent on a single scene —
useful for debugging, hyperparameter tuning, or quick proof-of-concept
runs.  It mirrors ``mariha-run-cl``'s registry/plugin pattern, so any
agent registered in ``mariha/rl/__init__.py`` is available here
automatically:

.. code-block:: bash

   mariha-run-single --agent sac --subject sub-01 --scene_id w1l1s0 --total_steps 200000
   mariha-run-single --agent ppo --subject sub-01 --scene_id w1l1s0 --total_steps 200000
   mariha-run-single --agent dqn --subject sub-01 --scene_id w1l1s0 --total_steps 200000

Under the hood, the agent is fed a ``StepBudgetCLEnv`` that cycles
forever over the scene's episode specs and flips ``is_done`` to ``True``
once the env-step budget is reached.  The budget is **soft** — agents
only check ``is_done`` at episode boundaries, so the actual step count
can overshoot by up to one episode length.

If ``--scene_id`` is omitted, the first scene in the subject's
curriculum is used.  ``--cl_method`` is honoured here too:

.. code-block:: bash

   mariha-run-single --agent dqn --cl_method ewc --scene_id w1l1s0

Live render checkpoints
-----------------------

Pass ``--render_every N`` to open a live window every *N* training episodes
and watch the agent play one full greedy episode.  Works for both
``mariha-run-single`` and ``mariha-run-cl`` (full CL curriculum):

.. code-block:: bash

   # Single-scene
   mariha-run-single --agent sac --subject sub-01 --scene_id w1l1s0 --render_every 50

   # Full CL run
   mariha-run-cl --agent sac --cl_method ewc --subject sub-01 --render_every 100

Training pauses, the window opens, the episode plays to termination, then
the window closes and training resumes.  The underlying implementation
(``play_render_episode`` in ``mariha.env.continual``) is shared between both
entry points.

Use ``--render_speed S`` together with ``--render_every`` to control
playback speed when a render window opens.  ``1.0`` is the native 60 fps,
values below ``1`` slow playback down (useful if the default is too fast
to follow), and values above ``1`` speed it up on a best-effort basis:

.. code-block:: bash

   mariha-run-cl --agent sac --cl_method ewc --subject sub-01 --render_every 100 --render_speed 0.5

Checkpointing
-------------

Per-task checkpoints are written by the runner at every task boundary
(and at the end of training) into:

.. code-block:: text

   experiments/checkpoints/{run_label}/{timestamp}_seed{seed}_task{k}/

where ``run_label`` is ``{agent}_{cl_method}`` if a CL method is
attached and ``{agent}`` otherwise (e.g. ``sac_ewc``, ``ppo``,
``dqn_der``), and ``k`` is the task index active when the checkpoint
was written.  Because every checkpoint write within task ``k`` uses the
same directory name, the directory always holds the *latest* weights
from that task — effectively one checkpoint per completed task.

This layout matches what ``mariha-evaluate --run_prefix
{timestamp}_seed{seed}`` discovers via
:func:`~mariha.eval.runner.find_task_checkpoints`, so the prefix string
the evaluator expects is exactly the run-dir leaf the runner emitted.
See :doc:`evaluation` for the full evaluation flow.
