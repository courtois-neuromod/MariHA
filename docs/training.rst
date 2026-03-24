Training
========

The :class:`~mariha.rl.sac.SAC` class implements the full episode-driven
training loop and serves as the base class for all CL methods.

Algorithm
---------

MariHA uses **Soft Actor-Critic (SAC)** adapted for discrete action spaces:

* The actor outputs logits over the 9-action space; a
  ``Categorical`` distribution is built from them.
* Entropy is the Shannon entropy of the categorical distribution.
* The critic outputs Q-values for *all* actions simultaneously
  (shape ``(batch, 9)``).
* The target entropy is ``0.98 × log(9) ≈ 2.09`` nats.
* Automatic entropy tuning: a per-task ``log_alpha`` variable is learned
  jointly with the policy.

Model architecture
------------------

Both actor and critic share the same CNN + MLP trunk design:

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

The CNN and dense trunk are the **common variables** — shared across tasks
and targeted by regularisation-based CL methods.  Task-specific heads (when
``num_heads > 1``) are excluded from regularisation.

Episode-driven training loop
-----------------------------

Unlike COOM's fixed step-budget loop, MariHA's loop runs until the
curriculum is exhausted:

.. code-block:: python

   while True:
       action = random_action  if step < start_steps else policy(obs)
       obs, r, done, info = env.step(action)
       replay_buffer.store(obs, action, r, next_obs, done, one_hot)

       if done:
           obs, info = env.reset()
           one_hot = info["task_one_hot"]
           if info["task_switch"]:
               on_task_end(old_task)
               _handle_task_change(new_task)   # resets buffer/opt/weights if configured
           if env.is_done:
               break

       if step >= update_after and step % update_every == 0:
           for _ in range(n_updates):
               batch          = replay_buffer.sample_batch(batch_size)
               episodic_batch = get_episodic_batch(task_idx)   # CL hook
               gradients, metrics = get_gradients(seq_idx, **batch)
               gradients = adjust_gradients(*gradients, episodic_batch=episodic_batch)  # CL hook
               apply_update(*gradients)

Task lifecycle
--------------

When ``info["task_switch"] = True`` is detected after a reset,
``_handle_task_change()`` is called:

1. ``on_task_start(new_task_idx)`` — CL hook; subclasses snapshot
   parameters, update importance weights, prune masks, fill episodic
   memory, etc.
2. Optionally reset the **replay buffer** (``reset_buffer_on_task_change``).
3. Optionally reset **actor / critic weights** (``reset_actor_on_task_change``).
4. Optionally reset the **optimizer** slot variables (``reset_optimizer_on_task_change``).
5. Recompile ``learn_on_batch`` via ``@tf.function`` — this specialises
   the graph for the new ``current_task_idx`` Python closure value.

CL extension points
--------------------

All CL methods extend :class:`~mariha.rl.sac.SAC` by overriding one or
more of the following methods:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Purpose
   * - ``get_auxiliary_loss(seq_idx)``
     - Return a scalar regularisation loss added to both actor and critic
       losses.  Called *inside* ``@tf.function``; must use TF ops only.
   * - ``adjust_gradients(actor_grads, critic_grads, alpha_grad, ...)``
     - Project or modify gradients after ``get_gradients()``.  Called
       inside ``@tf.function``; may create nested ``GradientTape``s.
   * - ``on_task_start(task_idx)``
     - Called once when a new task begins.  Runs *outside* TF graph
       compilation; numpy operations are safe.
   * - ``on_task_end(task_idx)``
     - Called once when a task ends.  Runs outside TF graph compilation.
   * - ``get_episodic_batch(task_idx)``
     - Return a dict of tensors sampled from episodic memory (used by
       AGEM, DER++, ClonEx).  Return ``None`` to disable.

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

Checkpointing
-------------

Actor and both critics (plus targets) are saved to:

.. code-block:: text

   experiments/checkpoints/{method}/{timestamp}_task{k}/

where ``k`` is the task index active when the checkpoint was written.
Because all checkpoint writes within task ``k`` use the same directory
name, the directory always holds the *latest* weights from that task —
effectively one checkpoint per completed task.

This layout is exploited by the evaluator to build the diagonal of the
CL performance matrix (see :doc:`evaluation`).
