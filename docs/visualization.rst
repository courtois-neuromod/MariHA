Visualization
=============

MariHA ships three scripts for turning training runs into watchable MP4 videos.
All scripts capture raw NES RGB frames (full colour, ~224 × 240 px) from the
emulator before the grayscale / resize observation pipeline, giving clean
game footage.

.. list-table:: Script comparison
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Script
     - Fidelity
     - Input required
     - Training flag needed
     - Videos produced
   * - ``replay_episode.py``
     - **Exact** (byte-identical)
     - trajectory ``.npz``
     - ``--save_trajectories`` (default on)
     - 1 per call
   * - ``visualize_episode.py``
     - **Approximate** (policy re-evaluated)
     - checkpoint dir
     - ``--save_weights``
     - 1 per call
   * - ``visualize_progress.py``
     - Exact *or* Approximate
     - run dir
     - trajectories or ``--checkpoint_every``
     - N (one per selected episode)

----

Exact replay — ``replay_episode.py``
--------------------------------------

Re-steps the environment with the *exact* action sequence logged during
training.  Because the same ``.state`` file (emulator RAM snapshot) is loaded
and the same actions are applied in order, the video is byte-for-byte
identical to what happened during training.

**Requirements:** ``--save_trajectories`` must have been enabled when
``mariha-run-single`` was run (this is the **default**).

**Usage:**

.. code-block:: bash

   python scripts/replay_episode.py \
       --trajectory experiments/single/w1l1s0/<run>/trajectories/episode_00050.npz \
       --output /tmp/ep50.mp4

.. list-table:: CLI flags
   :widths: 30 15 55
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--trajectory``
     - *(required)*
     - Path to an ``episode_NNNNN.npz`` file.
   * - ``--output``
     - ``replay.mp4``
     - Output MP4 file path.
   * - ``--fps``
     - ``60``
     - Frames per second (matches NES frame rate).

----

Weights-based replay — ``visualize_episode.py``
-------------------------------------------------

Loads saved actor weights and runs the agent greedily (argmax over logits) on
one episode.  This is **not** an exact replay — the stochastic policy may
choose different actions than it did during training — but is useful for
evaluating the trained agent on any scene, including scenes not seen during
training.

**Requirements:** run ``mariha-run-single`` with ``--save_weights`` to produce
a ``checkpoints/final/`` directory, or with ``--checkpoint_every N`` for
interval snapshots.

**Usage:**

.. code-block:: bash

   python scripts/visualize_episode.py \
       --subject sub-01 \
       --scene_id w1l1s0 \
       --checkpoint_dir experiments/single/w1l1s0/<run>/checkpoints/final \
       --output /tmp/viz.mp4

.. list-table:: CLI flags
   :widths: 30 15 55
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--subject``
     - ``sub-01``
     - Subject ID used to load an episode spec for the scene.
   * - ``--scene_id``
     - *(required)*
     - Scene to visualize (e.g. ``w1l1s0``).
   * - ``--checkpoint_dir``
     - *(required)*
     - Directory containing ``actor.index`` / ``actor.data-*`` files.
   * - ``--output``
     - ``visualize.mp4``
     - Output MP4 file path.
   * - ``--seed``
     - ``0``
     - RNG seed.
   * - ``--fps``
     - ``60``
     - Frames per second.

Architecture flags (``--hidden_sizes``, ``--activation``, ``--use_layer_norm``,
``--num_heads``, ``--hide_task_id``) must match those used during training.
The defaults match ``run_single.py`` defaults, so they only need to be set if
you changed them during training.

----

Progress visualization — ``visualize_progress.py``
----------------------------------------------------

Produces one MP4 per selected episode or checkpoint, letting you compare the
agent's behaviour at different points in training side-by-side.

Two modes:

* **trajectory** (default) — exact replay from saved action logs.
* **weights** — approximate replay from interval weight checkpoints.

**Usage — trajectory mode:**

.. code-block:: bash

   # Every 10th episode
   python scripts/visualize_progress.py \
       --run_dir experiments/single/w1l1s0/<run> \
       --every 10 \
       --output_dir /tmp/progress

   # Specific episodes
   python scripts/visualize_progress.py \
       --run_dir experiments/single/w1l1s0/<run> \
       --episodes 1,50,100,200 \
       --output_dir /tmp/progress

**Usage — weights mode:**

.. code-block:: bash

   python scripts/visualize_progress.py \
       --run_dir experiments/single/w1l1s0/<run> \
       --mode weights \
       --subject sub-01 \
       --scene_id w1l1s0 \
       --every 2 \
       --output_dir /tmp/progress

.. list-table:: CLI flags
   :widths: 30 15 55
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--run_dir``
     - *(required)*
     - Path to a ``mariha-run-single`` output directory.
   * - ``--mode``
     - ``trajectory``
     - ``trajectory`` or ``weights``.
   * - ``--episodes``
     - —
     - Comma-separated episode indices, e.g. ``1,50,100``.
       Mutually exclusive with ``--every``.
   * - ``--every``
     - —
     - Visualize every N episodes / checkpoints.
       Mutually exclusive with ``--episodes``.
   * - ``--output_dir``
     - ``<run_dir>/progress_videos/``
     - Directory for output MP4s.
   * - ``--fps``
     - ``60``
     - Frames per second.
   * - ``--subject``
     - ``sub-01``
     - *(weights mode only)* Subject for episode spec.
   * - ``--scene_id``
     - —
     - *(weights mode, required)* Scene to run the agent on.

Output files are named ``episode_NNNNN.mp4`` (trajectory mode) or
``step_NNNNNNN.mp4`` (weights mode).

----

``mariha.utils.video`` API
---------------------------

.. code-block:: python

   from mariha.utils.video import write_episode_video

   write_episode_video(frames, output_path, fps=60)

.. list-table:: Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``frames``
     - ``list[np.ndarray]``
     - List of ``(H, W, 3)`` uint8 RGB arrays, one per step.
       All frames must have the same shape.
   * - ``output_path``
     - ``str | Path``
     - Destination file path.  Parent directories are created automatically.
   * - ``fps``
     - ``int``
     - Frames per second.  Defaults to ``60`` (NES frame rate).

Uses ``cv2.VideoWriter`` with the ``mp4v`` codec.  Frames are converted from
RGB to BGR internally (OpenCV convention).

----

Determinism note
-----------------

``replay_episode.py`` and the trajectory mode of ``visualize_progress.py``
produce **deterministic** replays because:

1. The emulator is initialised from the exact ``.state`` file (gzip-compressed
   RAM snapshot) used during training.
2. The logged action sequence is replayed step by step — no policy inference
   takes place.

The same ``.state`` file plus the same actions always produce the same frames,
regardless of random seeds or TensorFlow version.

----

Future work
-----------

**CL progress visualization** (``visualize_cl_episode.py``)
    The full continual-learning training (``mariha-run``) already saves
    per-task actor weights in
    ``experiments/checkpoints/{method}/{timestamp}_task{k}/``.  A future script
    will load each task checkpoint and record the agent playing that task's
    scene, giving an approximate weights-based view of the CL agent at each
    task boundary.  Exact CL replay would require adding trajectory logging to
    ``mariha/rl/sac.py`` (shared by all 12 CL methods).
