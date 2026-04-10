Architecture
============

This page describes how MariHA's modules fit together, from raw gameplay
data to trained checkpoints and evaluation metrics.

.. note::
   MariHA is a port and extension of `COOM <https://github.com/TTomilin/COOM>`_,
   a continual RL benchmark built on VizDoom.  The original SAC training
   loop, model architecture, replay buffers, and most of the CL
   baselines were ported from COOM; MariHA then refactored the agents
   onto a shared :class:`~mariha.rl.base.BaseAgent` mixin so that SAC,
   PPO and DQN all share the same training loop, and converted the CL
   methods from SAC-only inheritance to agent-agnostic composition.

Data flow overview
------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │  data/mario.scenes/sub-XX/                                      │
   │  Human gameplay clips (.state + events TSV)                  │
   └──────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  mariha.curriculum                                              │
   │  HumanSequence → [EpisodeSpec, EpisodeSpec, ...]                │
   │  Ordered by clip_code; one EpisodeSpec per human clip           │
   └──────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  mariha.env.ContinualLearningEnv                                │
   │  Drives the SceneEnv stack through the episode sequence         │
   │  Emits task_switch=True when the scene ID changes               │
   └──────────────────────┬──────────────────────────────────────────┘
                          │  (obs, reward, done, info)
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Observation pipeline (per reset/step)                          │
   │  SceneEnv → ActionWrapper → GrayscaleWrapper →                  │
   │             ResizeWrapper → FrameStackWrapper → TaskIdWrapper   │
   │  Output: float32 (84,84,4) obs  +  info["task_one_hot"]        │
   └──────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  mariha.rl.base.TrainingLoopRunner   (shared training loop)     │
   │  Episode-driven: runs until ContinualLearningEnv.is_done        │
   │  Per-step  hooks: select_action → step → store_transition →    │
   │                   should_update → update_step                   │
   │  Per-task  hooks: on_task_end → on_task_change → on_task_start │
   │  Drives any BaseAgent: SAC, PPO, DQN, or a future MuZero/LLM    │
   └──────────────────────┬──────────────────────────────────────────┘
                          │  cl_method composition
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  mariha.methods.CLMethod  (composed onto agent.cl_method)       │
   │  on_task_start / on_task_end                                   │
   │  compute_loss_penalty (added inside the agent's GradientTape)   │
   │  adjust_gradients     (modify gradients before optimizer.step)  │
   │  after_gradient_step  (post-update hook for SI / PackNet)       │
   │  get_episodic_batch   (forwarded to adjust_gradients)           │
   └──────────────────────┬──────────────────────────────────────────┘
                          │  checkpoints
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  mariha.eval                                                    │
   │  eval_on_scene → R_final, R_diag                                │
   │  compute_cl_metrics → AP, BWT, forgetting, plasticity           │
   └─────────────────────────────────────────────────────────────────┘

Module responsibilities
-----------------------

``mariha.curriculum``
~~~~~~~~~~~~~~~~~~~~~

Reads BIDS-formatted events TSV files from ``data/mario.scenes`` and
converts them into an ordered list of :class:`~mariha.curriculum.episode.EpisodeSpec`
objects.  Each ``EpisodeSpec`` holds:

* The path to the ``.state`` emulator snapshot (starting game state)
* The frame budget (``max_steps = frame_stop - frame_start``)
* Scene and subject metadata

The :class:`~mariha.curriculum.sequences.HumanSequence` is a simple
iterator that yields specs in ``clip_code`` order — preserving the
chronological order of the human's play session.

``mariha.env``
~~~~~~~~~~~~~~

Three layers of environment abstraction:

1. :class:`~mariha.env.base.MarioEnv` — thin wrapper around ``stable-retro``
   that registers the bundled ``SuperMarioBros-Nes`` integration, loads
   ``.state`` files into the emulator, and computes the composite
   X-position reward (``Δx`` per step).

2. :class:`~mariha.env.scene.SceneEnv` — episode lifecycle manager.
   Enforces success / death / timeout termination conditions and
   accumulates :class:`~mariha.env.scene.EpisodeStats`.

3. :class:`~mariha.env.continual.ContinualLearningEnv` — sequences episodes
   across tasks.  Detects task boundaries and emits ``info["task_switch"]``
   and ``info["task_one_hot"]`` after each reset.

The observation **wrapper chain** (applied by
:func:`~mariha.env.continual.make_scene_env`) transforms raw ``(240, 256, 3)``
NES frames into the model input:

.. code-block:: text

   GrayscaleWrapper  →  (240, 256, 1)  uint8
   ResizeWrapper     →  (84, 84, 1)    uint8
   FrameStackWrapper →  (84, 84, 4)    uint8   (4-frame history)
   TaskIdWrapper     →  (84, 84, 4)    float32 ÷ 255  +  info["task_one_hot"]

``mariha.replay``
~~~~~~~~~~~~~~~~~

Buffer implementations sharing the same ``store`` / ``sample_batch``
interface.  The buffer type is selected at agent construction:

* **FIFO** (default) — standard ring buffer; overwrites oldest transitions.
* **Reservoir** — uniform over all transitions seen (not just recent).
* **Priority (SumTree)** — PER with ``SumTree``; classic implementation.
* **PER (SegmentTree)** — PER with ``SegmentTree``; cleaner IS weights.
* **EpisodicMemory** — fixed-capacity, never overwrites; used by AGEM
  and the distillation methods for storing past-task transitions with
  optional logit and Q-value targets.
* **RolloutBuffer** — on-policy storage with GAE advantages, used by
  PPO.
* **PerSceneBufferPool** — one replay buffer per scene with a
  session-boundary flush, used by per-scene buffer mode on SAC and DQN.

``mariha.rl``
~~~~~~~~~~~~~

The :mod:`mariha.rl.base` package contains the shared training-loop
infrastructure:

* :class:`~mariha.rl.base.BaseAgent` is a
  :class:`~mariha.benchmark.agent.BenchmarkAgent` subclass that
  implements ``run()`` once via :class:`~mariha.rl.base.TrainingLoopRunner`
  and exposes a small algorithm-agnostic callback surface
  (``select_action``, ``store_transition``, ``should_update``,
  ``update_step``, ``save_weights``, ``load_weights``).
* :class:`~mariha.rl.base.TrainingLoopRunner` is the episode-driven
  loop shared by per-step (off-policy) and per-rollout (on-policy)
  agents.  It owns episode bookkeeping, task-switch detection,
  session-boundary flushes, render checkpoints, periodic logging and
  saves.
* :func:`~mariha.rl.base.run_burn_in` is the shared burn-in driver
  used before the curriculum starts.

The three concrete agents — :class:`~mariha.rl.sac.SAC`,
:class:`~mariha.rl.ppo.PPO`, :class:`~mariha.rl.dqn.DQN` — subclass
:class:`~mariha.rl.base.BaseAgent` and implement only the
algorithm-specific pieces (network setup, loss formulation, gradient
step).  They additionally implement three agent-agnostic CL hooks:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Hook
     - Returns
   * - ``get_named_parameter_groups()``
     - ``dict[str, list[tf.Variable]]`` of logical parameter groups.
       SAC: ``{"actor", "critic1", "critic2"}``.  PPO: ``{"policy",
       "value"}``.  DQN: ``{"q"}``.
   * - ``forward_for_importance(obs, one_hot)``
     - One differentiable output tensor per group, used by EWC and
       MAS for Jacobian-based importance.
   * - ``distill_targets(obs, one_hot)``
     - Targets for distillation methods.  SAC/PPO return
       ``"actor_logits"``; DQN returns ``"q_values"``; ClonEx
       additionally consumes ``"critic1_q"``/``"critic2_q"`` (SAC) or
       ``"value"`` (PPO).

The :class:`~mariha.rl.models.MlpActor` and
:class:`~mariha.rl.models.MlpCritic` accept ``(obs, task_one_hot)`` as
separate inputs.  The CNN head (32→64→64 filters, Atari strides)
processes the pixel stack; its output is concatenated with the task
one-hot before the dense trunk.  PPO uses
:class:`~mariha.rl.models.PPOActorCritic`, which shares the conv trunk
between policy and value heads.

``mariha.methods``
~~~~~~~~~~~~~~~~~~

Continual learning methods are implemented as
:class:`~mariha.methods.base.CLMethod` subclasses that are *composed*
onto an agent rather than inherited from it.  Two intermediate base
classes share the bookkeeping for the major method families:

* :class:`~mariha.methods.regularizer_base.ParameterRegularizer` —
  the ``θ*`` snapshot, importance accumulation, and quadratic penalty
  shared by L2, EWC, MAS and SI.
* :class:`~mariha.methods.distillation_base.DistillationMethod` —
  the episodic memory and gradient injection shared by DER and ClonEx,
  with key-dispatched loss formulae for actor-logit KL (SAC/PPO),
  Q-value soft KL (DQN, Rusu 2016), and critic / value MSE
  (ClonEx).

See :doc:`cl_methods` for detailed descriptions of every baseline.

``mariha.eval``
~~~~~~~~~~~~~~~

Stateless utilities:

* :func:`~mariha.eval.runner.eval_on_scene` — builds a fresh env, runs the
  deterministic policy for ``n_episodes``, returns mean return +
  :class:`~mariha.env.scene.EpisodeStats` dicts.
* :func:`~mariha.eval.metrics.compute_cl_metrics` — computes AP, BWT,
  forgetting, and plasticity from ``R_final`` and ``R_diag`` arrays.
* :func:`~mariha.eval.runner.find_task_checkpoints` — discovers per-task
  checkpoint directories by glob.

Relationship to COOM
--------------------

MariHA is architecturally derived from
`COOM <https://github.com/TTomilin/COOM>`_ (Tomilin et al., 2023), a
continual RL benchmark built on VizDoom.  The following components are
ported (with import-path changes and VizDoom/LSTM references removed):

* SAC training loop and the conceptual catalogue of CL baselines
  (L2, EWC, MAS, A-GEM, PackNet, ClonEx, MultiTask).
* Actor / Critic CNN architecture (``mariha.rl.models``).
* Replay buffers and priority trees (``mariha.replay``).

MariHA introduces the following extensions over COOM:

* **Human-aligned curriculum** — episode specs from real player sessions
  replace fixed step-budget training.
* **PPO and DQN agents** alongside SAC, all sharing
  :class:`~mariha.rl.base.BaseAgent` and the same training loop.
* **Agent-agnostic CL composition** — every CL baseline runs on every
  RL agent through the
  :meth:`~mariha.rl.base.BaseAgent.get_named_parameter_groups` /
  :meth:`~mariha.rl.base.BaseAgent.forward_for_importance` /
  :meth:`~mariha.rl.base.BaseAgent.distill_targets` hooks, instead of
  inheriting from a SAC-only base class.
* **Synaptic Intelligence (SI)** — online importance accumulation
  (Zenke et al., 2017); not present in COOM.
* **Dark Experience Replay (DER)** — actor / Q-value distillation from
  an episodic memory (Buzzega et al., 2020).
* **DQN-side distillation** — the
  :class:`~mariha.methods.distillation_base.DistillationMethod` base
  dispatches to soft-KL on Q-values for DQN (Rusu et al., 2016,
  *Policy Distillation*), so DER and ClonEx work on DQN as well.
* **Episode-driven loop** — training ends when the curriculum is exhausted,
  not at a fixed global step count.
* **Evaluation suite** — ``mariha.eval`` with AP, BWT, forgetting,
  plasticity, and Mario-specific behavioral metrics (clear rate,
  x_traveled, score, deaths).
