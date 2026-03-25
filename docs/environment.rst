Environment
===========

The environment stack transforms raw emulator frames and human clip
metadata into the observation tensors and task vectors consumed by the
SAC agent.

Layer overview
--------------

.. code-block:: text

   MarioEnv                          ← stable-retro + state loading
   └── SceneEnv                      ← episode lifecycle, stats, termination
       └── ActionWrapper             ← maps discrete 0–8 → NES button array
           └── GrayscaleWrapper      ← converts RGB→greyscale
               └── ResizeWrapper     ← downsamples to 84×84
                   └── FrameStackWrapper    ← stacks 4 consecutive frames
                       └── TaskIdWrapper    ← adds task_one_hot to info on reset
                           └── ContinualLearningEnv  ← sequences episodes

Each layer is applied by :func:`~mariha.env.continual.make_scene_env` and
can be imported from :mod:`mariha.env.wrappers`.

Layers in detail
----------------

``MarioEnv``
~~~~~~~~~~~~

:class:`~mariha.env.base.MarioEnv` registers the bundled
``SuperMarioBros-Nes`` integration from
``data/mario/stimuli/SuperMarioBros-Nes/`` via
``retro.data.Integrations.add_custom_path()``.  This avoids any
system-wide ROM installation — the integration is fully self-contained.

On each ``step()``, the raw ``stable-retro`` reward is discarded and
replaced by ``Δx_position`` (the change in Mario's absolute X-coordinate
since the previous step).  This gives a dense, shaped reward signal that
encourages forward progress.

``SceneEnv``
~~~~~~~~~~~~

:class:`~mariha.env.scene.SceneEnv` wraps ``MarioEnv`` and manages the
episode lifecycle:

* ``reset(episode_spec)`` loads the human clip's ``.state`` file,
  snapshots the initial lives, score, coins, and X-position.
* ``step()`` checks three termination conditions after each step:

  * **Success** (``terminated=True``): ``x_pos ≥ exit_point``
  * **Death** (``terminated=True``): ``lives < initial_lives``
  * **Timeout** (``truncated=True``): ``step_count ≥ max_steps``

After the terminal step, :class:`~mariha.env.scene.EpisodeStats` is
populated with ``cleared``, ``x_traveled``, ``score_gained``,
``coins_gained``, ``lives_lost``, and ``outcome``.

``ActionWrapper``
~~~~~~~~~~~~~~~~~

Mario NES has 9 relevant button combinations.  The
:class:`~mariha.env.wrappers.action.ActionWrapper` maps a discrete integer
0–8 to the corresponding ``MultiBinary`` NES button array, reducing the
action space from 2⁹=512 to 9.

Observation wrappers
~~~~~~~~~~~~~~~~~~~~

* **GrayscaleWrapper** — converts the ``(240, 256, 3)`` RGB frame to
  greyscale ``(240, 256, 1)`` using a weighted average (standard
  luminance formula).
* **ResizeWrapper** — downsamples to ``(84, 84, 1)`` using bilinear
  interpolation (OpenCV).
* **FrameStackWrapper** — maintains a circular buffer of the 4 most
  recent frames and stacks them along the channel axis:
  ``(84, 84, 4)`` uint8.

``TaskIdWrapper``
~~~~~~~~~~~~~~~~~

:class:`~mariha.env.wrappers.observation.TaskIdWrapper` applies two
transformations:

1. **Normalisation**: converts the uint8 ``(84, 84, 4)`` stack to
   float32 ``[0, 1]`` by dividing by 255.
2. **Task one-hot**: on every ``reset()``, adds
   ``info["task_one_hot"]`` — a float32 vector of length ``num_tasks``
   with a 1 at the index of the current scene in the canonical scene
   list.

The one-hot is NOT concatenated to the observation — it is passed as a
separate second input to the actor and critic models, following the
COOM two-input convention.

``ContinualLearningEnv``
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mariha.env.continual.ContinualLearningEnv` sequences episodes
from the curriculum:

* Each ``reset()`` call pops the next ``EpisodeSpec`` from the sequence.
* If the scene changed, the old env is closed and a new one is built for
  the new scene; ``info["task_switch"] = True`` is set.
* ``env.is_done`` becomes ``True`` when the sequence is exhausted.

The SAC training loop polls ``env.is_done`` after each episode to detect
the end of training.

Using the environment directly
-------------------------------

.. code-block:: python

   from mariha.curriculum.sequences import HumanSequence
   from mariha.env.continual import ContinualLearningEnv
   from mariha.env.scenario_gen import load_metadata, SCENARIOS_DIR

   scene_meta = load_metadata(SCENARIOS_DIR)
   scene_ids  = sorted(scene_meta.keys())
   sequence   = HumanSequence(subject_id="sub-01")
   env        = ContinualLearningEnv(sequence=sequence, scene_ids=scene_ids)

   obs, info = env.reset()
   print(obs.shape)           # (84, 84, 4)
   print(info["task_one_hot"].shape)  # (313,)

   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Building a single-scene env (for eval)
---------------------------------------

.. code-block:: python

   from mariha.env.continual import make_scene_env

   env = make_scene_env(
       scene_id   = "w1l1s0",
       exit_point = scene_meta["w1l1s0"]["exit_point"],
       scene_ids  = scene_ids,
   )
   obs, info = env.reset(episode_spec=spec)
