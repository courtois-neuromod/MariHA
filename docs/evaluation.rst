Evaluation
==========

MariHA's evaluation suite measures both standard CL metrics and
Mario-specific behavioral metrics.

Protocol
--------

Evaluation uses the **same scenes as training**.  For each unique scene in
the subject's curriculum, the first ``EpisodeSpec`` is used as the
canonical eval episode — the same starting state file and frame budget
that were used during training.

Two evaluation modes are available:

**Final-only** (default, fast)
    Load the final checkpoint (highest task index found), run ``n_episodes``
    per scene, compute AP and behavioral metrics.
    ~1600 episodes for 313 scenes × 5 episodes.

**Diagonal** (``--eval_diagonal``, slower)
    Additionally load the per-task checkpoint for each scene and evaluate
    it on that scene.  Enables BWT, forgetting, and plasticity metrics.
    ~3200 episodes total for 313 scenes.

Running evaluation
------------------

.. code-block:: bash

   mariha-evaluate \
     --subject     sub-01 \
     --cl_method   ewc \
     --run_prefix  20260322_120000_seed0 \
     --n_episodes  5 \
     --eval_diagonal

The ``run_prefix`` is the timestamp + seed string from the training run
(printed to the console at training start and used to name log
directories).

Output
------

Results are saved to
``experiments/{subject}/{method}/{run_prefix}/eval_results.json``:

.. code-block:: json

   {
     "metadata": {
       "subject": "sub-01",
       "cl_method": "ewc",
       "n_episodes": 5,
       "n_scenes": 313,
       "eval_diagonal": true
     },
     "cl_metrics": {
       "AP":            12.4,
       "AP_std":         8.1,
       "BWT":           -3.2,
       "forgetting":     3.2,
       "plasticity":    18.6,
       "per_scene_final": { "w1l1s0": 15.3, ... },
       "per_scene_peak":  { "w1l1s0": 17.1, ... }
     },
     "behavioral_summary": {
       "clear_rate":        0.42,
       "mean_x_traveled":  210.3,
       "mean_score_gained":  840,
       "death_rate":         0.31
     },
     "per_scene_behavioral": { ... }
   }

Metric definitions
------------------

CL metrics
~~~~~~~~~~

Let :math:`R[i][j]` be the return on scene :math:`j` evaluated with the
checkpoint saved after training on task :math:`i`.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Metric
     - Formula
   * - **AP** (Average Performance)
     - :math:`\frac{1}{N} \sum_j R[\text{final}][j]` — mean return across
       all scenes with the final model.  The primary summary metric.
   * - **BWT** (Backward Transfer)
     - :math:`\frac{1}{N-1} \sum_{j<N} \bigl(R[\text{final}][j] - R[j][j]\bigr)`
       — how final performance differs from peak performance on past tasks.
       Negative = forgetting; positive = positive backward transfer (rare).
   * - **Forgetting**
     - :math:`\max(0, -\text{BWT})` — unsigned catastrophic forgetting score.
   * - **Plasticity**
     - :math:`R[N][N]` — performance on the last task right after training
       on it.  Measures the agent's ability to learn new tasks without
       interference.

.. note::
   BWT and forgetting require ``--eval_diagonal``.  Without it, only AP,
   AP_std, and per-scene final performance are reported.

Behavioral metrics
~~~~~~~~~~~~~~~~~~

Derived from :class:`~mariha.env.scene.EpisodeStats` (collected during
eval rollouts):

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Metric
     - Description
   * - **clear_rate**
     - Fraction of episodes where Mario reached the scene exit point.
       The most interpretable performance metric.
   * - **mean_x_traveled**
     - Mean absolute X-distance covered per episode.  Partial credit
       for how far the agent progressed even without completing the scene.
   * - **mean_score_gained**
     - Mean game score delta per episode.
   * - **death_rate**
     - Fraction of episodes ending in Mario's death (lives lost).

Relationship to CL performance matrix
---------------------------------------

The full CL performance matrix :math:`R[i][j]` (``N × N`` for ``N``
scenes) is never fully computed — for 313 scenes this would require
313 × 313 × 5 = ~490k evaluation episodes.  Instead:

* The **final row** :math:`R[\text{final}][j]` is computed for all ``j``
  using the final checkpoint (``final_only`` mode).
* The **diagonal** :math:`R[j][j]` is computed by loading the per-task
  checkpoint ``task_j`` and evaluating on scene ``j`` only
  (``diagonal`` mode).

This yields AP, BWT, and forgetting without evaluating the full matrix.
