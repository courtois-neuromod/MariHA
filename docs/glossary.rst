Glossary
========

MariHA mixes vocabulary from four domains — reinforcement learning,
continual learning, neuroimaging (BIDS), and game design — and the same
real-world thing has multiple plausible names depending on which domain
you walked in from.  This page fixes the canonical term per concept and
explains where each one is allowed in code.

Canonical terms
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Term
     - Domain
     - Meaning in MariHA
   * - ``run`` / ``run_id``
     - Neuroimaging (BIDS)
     - One BIDS fMRI run = one ``*_desc-scenes_events.tsv`` file.  84–102
       per subject, split into discovery (~28%) and practice (~72%).
       This is the unit of a continual-learning **task**.
   * - ``run_index``
     - Neuroimaging
     - Chronological 0-based ordinal of a run within a subject's
       curriculum.  Equal to the integer ``task_idx`` seen by the agent.
   * - ``scene`` / ``scene_id``
     - Game design
     - One playable Mario sub-region (e.g. ``w1l1s0``).  Drives the
       emulator state file.  Scenes are visited many times across the
       curriculum and overlap freely across runs.
   * - ``level``
     - Game design
     - Mario level (e.g. ``w1l1``).  A scene belongs to exactly one
       level; a run may contain one level (discovery) or several
       (practice).
   * - ``clip``
     - MariHA-specific (curriculum data)
     - One human attempt at one scene.  Data unit of the curriculum: one
       :class:`~mariha.curriculum.episode.EpisodeSpec` = one clip.
       Identified by a 14-character ``clip_code``.  ~13k per subject.
   * - ``episode``
     - RL
     - One agent rollout in the environment.  In MariHA training, one
       clip drives one episode (the ``EpisodeSpec`` defines the start
       state and step budget).  Don't use "episode" for human data —
       that's a clip.
   * - ``session``
     - Neuroimaging (BIDS)
     - One scanning session (``ses-001``, etc.).  Contains multiple
       runs.  Surfaces in ``info["session"]`` mainly for replay-buffer
       flush logic; not a CL task boundary.
   * - ``subject``
     - Neuroimaging (BIDS)
     - One human participant (``sub-01``, etc.).
   * - ``task`` / ``task_idx``
     - Continual learning (abstract)
     - **Layer-restricted**: the abstract integer index used inside the
       agent / CL framework only.  ``task_idx == run_index`` for this
       benchmark; the agent layer doesn't need to know.  See *Layering*
       below.
   * - ``run_label``
     - MariHA-specific (experiment metadata)
     - The directory name combining agent, CL method, timestamp and
       seed (e.g. ``sac_ewc/20260322_120000_seed0``).  Identifies one
       experiment, **not** a BIDS run.  Avoid the bare word "run" for
       this concept.

Layering
--------

The vocabulary is layer-restricted to keep modules decoupled:

- **Curriculum and environment layers** (``mariha.curriculum``,
  ``mariha.env``) speak in domain-native terms: ``run_id``, ``scene_id``,
  ``clip_code``, ``session``.  The word *task* is forbidden here.
- **Agent and CL layers** (``mariha.rl.base``, ``mariha.methods``) speak
  in CL-abstract terms: ``task_idx``, ``num_tasks``, hooks named
  ``on_task_start`` / ``on_task_end``.  The agent doesn't know that a
  task happens to be a BIDS run.
- **Translation** happens at exactly one place — the training loop
  (``mariha.rl.base.training_loop``) and the eval pipeline
  (``mariha.eval``) — which read ``info["run_id"]`` from the env and
  pass ``run_ids.index(run_id)`` to the agent as ``task_idx``.

This means the CL framework can be retargeted at a different task
definition (sessions, phases, levels, …) by changing one mapping in the
config layer, without touching agents or CL methods.

Forbidden / deprecated terms
----------------------------

- **subtask** — was used as a synonym for "scene within a run" during
  the run-as-task refactor.  Now removed; say ``scene`` instead.  The
  env emits ``info["scene_switch"]`` (not ``subtask_switch``) when the
  scene changes within a run.
- **training run** as a colloquial code identifier — use
  ``run_label`` for the directory and ``experiment`` in prose.  Bare
  "run" in code always means BIDS run.
- **stage** — never used.  Don't introduce it.

The ``info`` dict
-----------------

Every ``ContinualLearningEnv.reset()`` advances exactly one clip — the
clip is the atomic unit of the curriculum.  The ``*_switch`` flags below
are *conditions on that transition*: they report whether the run, the
scene, or the session differs from the previous clip's.

The wrappers around ``SceneEnv`` populate ``info`` with these canonical
keys (one per row of the table above where applicable):

- ``run_id`` (str), ``run_index`` (int), ``task_one_hot`` (vector of
  length ``len(run_ids)``)
- ``scene_id`` (str)
- ``session`` (str)
- ``task_switch`` (bool) — a new BIDS run starts.  Drives CL hooks.
- ``scene_switch`` (bool) — the scene changes within a run.  Logged
  only; no CL hook fires.
- ``session_switch`` (bool) — a new BIDS session starts.  Drives
  per-scene buffer flushes.
