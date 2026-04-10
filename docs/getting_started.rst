Getting Started
===============

Requirements
------------

* Python 3.9+
* The ``mario.scenes`` dataset (git-annex / DataLad)
* CUDA-capable GPU recommended for training (CPU works but is slow)

Installation
------------

Clone the repository and run the setup script:

.. code-block:: bash

   git clone https://github.com/courtois-neuromod/mariha MariHA
   cd MariHA
   bash setup.sh
   source env/bin/activate

``setup.sh`` will:

1. Create a virtual environment at ``env/``
2. Install the package and all dependencies (including ``tf_keras`` for
   TensorFlow ≥ 2.16 compatibility)
3. Generate the per-scene ``scenario.json`` files
4. Smoke-test the installation

Pulling the dataset
-------------------

Subject gameplay data is stored in ``data/mario.scenes`` as a
`DataLad <https://www.datalad.org>`_ / git-annex dataset:

.. code-block:: bash

   # All subjects
   cd data/mario.scenes && git annex get sub-*/

   # Single subject
   cd data/mario.scenes && git annex get sub-01/

Each subject's data contains ``.state`` files — emulator snapshots that
place Mario at exactly the position and game state a human player was at
during a recorded clip.

Training — full CL curriculum
------------------------------

.. code-block:: bash

   mariha-run-cl \
     --algorithm  ewc \
     --subject    sub-01 \
     --seed       0

This trains the agent on the full ordered sequence of scenes from subject
``sub-01``'s play history.  Checkpoints are saved to
``experiments/checkpoints/ewc/``.

Key training arguments:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--subject``
     - *(required)*
     - Subject ID: ``sub-01``, ``sub-02``, ``sub-03``, ``sub-05``, ``sub-06``
   * - ``--algorithm``
     - ``sac``
     - Algorithm name (see :doc:`cl_methods`).  Supports ``sac``, ``ppo``, ``dqn``, and all CL methods.
   * - ``--seed``
     - ``0``
     - Global random seed.
   * - ``--batch_size``
     - ``128``
     - Replay-buffer mini-batch size.
   * - ``--replay_size``
     - ``100000``
     - Replay buffer capacity.
   * - ``--lr``
     - ``1e-3``
     - Initial learning rate.
   * - ``--n_updates``
     - ``50``
     - Gradient steps per update round.
   * - ``--render_every``
     - ``0``
     - Open a live window every this many episodes to watch the agent play
       one full greedy episode. ``0`` = disabled.
   * - ``--render_speed``
     - ``1.0``
     - Playback speed multiplier for the live render window. ``1.0`` = 60 fps
       (native), ``<1`` slows playback down, ``>1`` speeds it up (best-effort).

Training — single scene (debugging)
-------------------------------------

.. code-block:: bash

   mariha-run-single --scene_id w1l1s0 --seed 0

Pass ``--render_every N`` to open a live window and watch one full greedy
episode every *N* training episodes — useful for checking progress at a
glance without slowing down training.  Combine it with ``--render_speed S``
to slow down or speed up playback (``1.0`` = 60 fps native):

.. code-block:: bash

   mariha-run-single --scene_id w1l1s0 --seed 0 --render_every 50 --render_speed 0.5

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--total_steps``
     - ``200000``
     - Total environment steps for single-scene training.
   * - ``--render_every``
     - ``0``
     - If > 0, open a live window every this many episodes to watch the
       agent play one full greedy episode. ``0`` = disabled.
   * - ``--render_speed``
     - ``1.0``
     - Playback speed multiplier for the live render window. ``1.0`` = 60 fps
       (native), ``<1`` slows playback down, ``>1`` speeds it up (best-effort).

Evaluation
----------

After training, evaluate the saved checkpoints:

.. code-block:: bash

   mariha-evaluate \
     --subject     sub-01 \
     --algorithm   ewc \
     --run_prefix  20260322_120000_seed0 \
     --n_episodes  5 \
     --eval_diagonal

The ``--eval_diagonal`` flag loads the per-task checkpoint for each scene
and evaluates it on that scene, enabling BWT and forgetting computation.
Without it, only the final checkpoint is used (faster).

Results are saved to
``experiments/sub-01/ewc/20260322_120000_seed0/eval_results.json``.

See :doc:`evaluation` for a full description of all reported metrics.
