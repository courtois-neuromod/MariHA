Architecture
============

This page describes how MariHA's modules fit together, from raw gameplay
data to trained checkpoints and evaluation metrics.

.. note::
   MariHA is a port and extension of `COOM <https://github.com/TTomilin/COOM>`_,
   a continual RL benchmark built on VizDoom.  The SAC training loop,
   model architecture, replay buffers, and 8 of the 11 CL baselines
   follow COOM's design; MariHA adapts them to the Mario domain, adds a
   human-aligned episode curriculum, and introduces three new methods
   (SI, DER++, OWL).

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
   │  mariha.rl.SAC  (training loop)                                 │
   │  Episode-driven: runs until ContinualLearningEnv.is_done        │
   │  On task_switch: calls on_task_end / _handle_task_change        │
   │  Every update_every steps: learn_on_batch → gradients →         │
   │                            adjust_gradients → apply_update      │
   └──────────────────────┬──────────────────────────────────────────┘
                          │  gradient hooks
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  mariha.methods.*  (CL baseline)                                │
   │  Overrides: adjust_gradients / get_auxiliary_loss /             │
   │             on_task_start / on_task_end / get_episodic_batch    │
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

Five buffer implementations sharing the same ``store`` / ``sample_batch``
interface.  The buffer type is selected at agent construction:

* **FIFO** (default) — standard ring buffer; overwrites oldest transitions.
* **Reservoir** — uniform over all transitions seen (not just recent).
* **Priority (SumTree)** — PER with ``SumTree``; classic implementation.
* **PER (SegmentTree)** — PER with ``SegmentTree``; cleaner IS weights.
* **EpisodicMemory** — fixed-capacity, never overwrites; used by AGEM,
  DER++, ClonEx for storing past-task transitions with optional logit
  and Q-value targets.

``mariha.rl``
~~~~~~~~~~~~~

The :class:`~mariha.rl.sac.SAC` class contains the full training loop and
exposes five hook methods that CL subclasses override:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Hook
     - Called when
   * - ``on_task_start(task_idx)``
     - First reset after a task switch; before any learning on the new task.
   * - ``on_task_end(task_idx)``
     - When ``task_switch=True`` is detected in ``info`` (task just finished).
   * - ``get_auxiliary_loss(seq_idx)``
     - Inside every ``get_gradients()`` call; return a scalar regularisation loss.
   * - ``adjust_gradients(...)``
     - After ``get_gradients()``; modify or project gradients before the update.
   * - ``get_episodic_batch(task_idx)``
     - Before each ``learn_on_batch()``; supply a past-task replay batch.

The :class:`~mariha.rl.models.MlpActor` and :class:`~mariha.rl.models.MlpCritic`
both accept ``(obs, task_one_hot)`` as separate inputs.  The CNN head
(32→64→64 filters, Atari strides) processes the pixel stack; its output
is concatenated with the task one-hot before the dense trunk.

``mariha.methods``
~~~~~~~~~~~~~~~~~~

All 12 CL baselines inherit from :class:`~mariha.rl.sac.SAC` (or from
:class:`~mariha.methods.regularization.Regularization_SAC` for the
regularisation family).  See :doc:`cl_methods` for detailed descriptions.

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
ported directly (with import-path changes and VizDoom/LSTM references
removed):

* SAC training loop (``mariha.rl.sac``)
* Actor / Critic CNN architecture (``mariha.rl.models``)
* Replay buffers and priority trees (``mariha.replay``)
* L2, EWC, MAS, PackNet, A-GEM, VCL, ClonEx, MultiTask baselines

MariHA introduces the following extensions over COOM:

* **Human-aligned curriculum** — episode specs from real player sessions
  replace fixed step-budget training.
* **Synaptic Intelligence (SI)** — online importance accumulation (Zenke
  et al., 2017); not present in COOM.
* **DER++** — actor distillation from episodic memory (Buzzega et al.,
  2020); not present in COOM.
* **OWL** — multi-head SAC with EWC on the shared trunk and an EWAF
  bandit for test-time task inference (Kessler et al., 2022);
  not present in COOM.
* **Episode-driven loop** — training ends when the curriculum is exhausted,
  not at a fixed global step count.
* **Evaluation suite** — ``mariha.eval`` with AP, BWT, forgetting,
  plasticity, and Mario-specific behavioral metrics (clear rate,
  x_traveled, score, deaths).
