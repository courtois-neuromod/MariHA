mariha.rl
=========

The RL package contains the three built-in agents (SAC, PPO, DQN), the
shared :class:`~mariha.rl.base.BaseAgent` infrastructure that owns the
training loop once for all of them, and the neural network architectures
they share.

.. seealso:: :doc:`/training` — algorithm description, training-loop walkthrough, and CL composition points.

mariha.rl.base
--------------

Shared training-loop infrastructure.  :class:`~mariha.rl.base.BaseAgent`
implements the :class:`~mariha.benchmark.agent.BenchmarkAgent` contract
in terms of a small callback surface (``select_action``,
``store_transition``, ``should_update``, ``update_step``,
``save_weights``, ``load_weights``); :class:`~mariha.rl.base.TrainingLoopRunner`
provides the episode-driven loop shared by per-step and per-rollout
agents.

.. automodule:: mariha.rl.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mariha.rl.base.agent_base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mariha.rl.base.training_loop
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mariha.rl.base.burn_in
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mariha.rl.base.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

mariha.rl.sac
-------------

.. automodule:: mariha.rl.sac
   :members:
   :undoc-members:
   :show-inheritance:

mariha.rl.ppo
-------------

.. automodule:: mariha.rl.ppo
   :members:
   :undoc-members:
   :show-inheritance:

mariha.rl.dqn
-------------

.. automodule:: mariha.rl.dqn
   :members:
   :undoc-members:
   :show-inheritance:

mariha.rl.random.agent
----------------------

.. automodule:: mariha.rl.random.agent
   :members:
   :undoc-members:
   :show-inheritance:

mariha.rl.models
----------------

.. automodule:: mariha.rl.models
   :members:
   :undoc-members:
   :show-inheritance:
