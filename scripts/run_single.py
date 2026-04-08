"""Single-scene SAC training script (development / debugging).

Trains vanilla SAC on a single Mario scene drawn from the human curriculum.
Useful for verifying the environment, tuning hyperparameters, or quick
proof-of-concept runs before the full continual-learning experiment.

Usage::

    python scripts/run_single.py --subject sub-01 --scene_id w1l1s0

Or via the entry point::

    mariha-run-single --subject sub-01 --scene_id w1l1s0

The number of training steps defaults to 200 000 (``--total_steps``).
If ``--scene_id`` is not provided, the first scene in the subject's
curriculum is used.

Pass ``--render_every N`` to open a live window and watch one full greedy
episode every N training episodes (a progress checkpoint)::

    mariha-run-single --subject sub-01 --scene_id w1l1s0 --render_every 50
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools

import numpy as np
import tensorflow as tf

from mariha.curriculum.loader import load_curriculum
from mariha.env.continual import make_scene_env
from mariha.env.scenario_gen import load_metadata
from mariha.env.base import SCENARIOS_DIR, STIMULI_PATH
from mariha.replay.buffers import BufferType, ReplayBuffer
from mariha.rl import models
from mariha.rl.sac import SAC
from mariha.utils.config import build_parser
from mariha.utils.logging import EpochLogger
from mariha.utils.running import get_activation_from_str, get_readable_timestamp


def render_episode(actor, env_kwargs: dict, spec, one_hot) -> None:
    """Spin up a human-render env, play one greedy episode, then close it."""
    from tensorflow_probability.python.distributions import Categorical

    render_env = make_scene_env(**env_kwargs, render_mode="human")
    obs, _ = render_env.reset(episode_spec=spec)
    done = False
    while not done:
        logits = actor(
            tf.expand_dims(tf.convert_to_tensor(obs), 0),
            tf.expand_dims(tf.convert_to_tensor(one_hot), 0),
        )
        action = int(Categorical(logits=logits).mode().numpy()[0])
        obs, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
    render_env.close()


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--total_steps", type=int, default=200_000,
        help="Total environment steps for single-scene training."
    )
    parser.add_argument(
        "--render_every", type=int, default=0,
        help=(
            "If > 0, open a live window and play one full greedy episode every "
            "this many training episodes (a checkpoint render). 0 = disabled."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve the scene to train on
    # ------------------------------------------------------------------ #
    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())

    if args.scene_id is not None:
        scene_id = args.scene_id
    else:
        # Use the first scene in the subject's curriculum.
        specs = load_curriculum(
            subject_id=args.subject,
            require_existing_states=args.require_existing_states,
        )
        if not specs:
            print(f"ERROR: No episodes found for {args.subject}.")
            sys.exit(1)
        scene_id = specs[0].scene_id

    if scene_id not in scene_meta:
        print(f"ERROR: scene_id '{scene_id}' not found in scenarios metadata.")
        sys.exit(1)

    # Load episode specs for this scene so SceneEnv.reset() gets a state file.
    all_specs = load_curriculum(
        subject_id=args.subject,
        require_existing_states=args.require_existing_states,
    )
    scene_specs = [s for s in all_specs if s.scene_id == scene_id]
    if not scene_specs:
        print(
            f"ERROR: No episode specs found for scene '{scene_id}' "
            f"(subject={args.subject}). "
            "Check that subject data is available and state files exist."
        )
        sys.exit(1)
    spec_cycle = itertools.cycle(scene_specs)

    print(f"Training on scene: {scene_id}  ({len(scene_specs)} episode spec(s))")

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    exit_point = scene_meta[scene_id]["exit_point"]
    env_kwargs = dict(
        scene_id=scene_id,
        exit_point=exit_point,
        scene_ids=scene_ids,
    )
    env = make_scene_env(**env_kwargs, render_mode=args.render_mode)

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #
    timestamp = get_readable_timestamp()
    experiment_dir = Path(args.experiment_dir)
    run_dir = str(experiment_dir / "single" / scene_id / f"{timestamp}_seed{args.seed}")
    logger = EpochLogger(
        output_dir=run_dir,
        logger_output=args.logger_output,
        config=vars(args),
        group_id=f"single_{scene_id}",
    )

    # ------------------------------------------------------------------ #
    # Build a minimal single-scene training loop
    # ------------------------------------------------------------------ #
    activation = get_activation_from_str(args.activation)
    policy_kwargs = dict(
        state_space=env.observation_space,
        action_space=env.action_space,
        num_tasks=len(scene_ids),
        hidden_sizes=tuple(args.hidden_sizes),
        activation=activation,
        use_layer_norm=args.use_layer_norm,
        num_heads=args.num_heads,
        hide_task_id=args.hide_task_id,
    )

    actor = models.MlpActor(**policy_kwargs)
    critic1 = models.MlpCritic(**policy_kwargs)
    target_critic1 = models.MlpCritic(**policy_kwargs)
    target_critic1.set_weights(critic1.get_weights())
    critic2 = models.MlpCritic(**policy_kwargs)
    target_critic2 = models.MlpCritic(**policy_kwargs)
    target_critic2.set_weights(critic2.get_weights())

    replay = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        size=args.replay_size,
        num_tasks=len(scene_ids),
    )

    # one_hot for this scene
    task_idx = scene_ids.index(scene_id)
    one_hot = np.zeros(len(scene_ids), dtype=np.float32)
    one_hot[task_idx] = 1.0

    try:
        from tf_keras.optimizers import Adam
    except ImportError:
        from tensorflow.keras.optimizers import Adam  # type: ignore[no-redef]
    from tensorflow_probability.python.distributions import Categorical
    import time
    from tqdm import tqdm

    actor_optimizer = Adam(learning_rate=args.lr)
    critic_optimizer = Adam(learning_rate=args.lr)
    gamma = args.gamma
    polyak = args.polyak
    batch_size = args.batch_size
    act_dim = env.action_space.n

    obs, info = env.reset(episode_spec=next(spec_cycle))
    episodes = 0
    ep_return = 0.0
    ep_len = 0
    start_time = time.time()

    pbar = tqdm(total=args.total_steps, unit="step", dynamic_ncols=True)
    for step in range(args.total_steps):
        # Action
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            logits = actor(
                tf.expand_dims(tf.convert_to_tensor(obs), 0),
                tf.expand_dims(tf.convert_to_tensor(one_hot), 0),
            )
            action = int(Categorical(logits=logits).sample().numpy()[0])

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += reward
        ep_len += 1

        replay.store(obs, action, reward, next_obs, terminated, one_hot)
        obs = next_obs

        if done:
            episodes += 1
            logger.store({"train/return": ep_return, "train/ep_length": ep_len})
            pbar.set_postfix(ep=episodes, ret=f"{ep_return:.1f}", refresh=False)
            ep_return = 0.0
            ep_len = 0

            if args.render_every > 0 and episodes % args.render_every == 0:
                pbar.write(f"[render] episode {episodes} — opening live window...")
                env.close()
                render_episode(actor, env_kwargs, next(spec_cycle), one_hot)
                env = make_scene_env(**env_kwargs, render_mode=args.render_mode)

            obs, info = env.reset(episode_spec=next(spec_cycle))

        # Policy update
        if step >= args.update_after and step % args.update_every == 0 and replay.size >= batch_size:
            for _ in range(args.n_updates):
                b = replay.sample_batch(batch_size)
                obs_t = b["obs"]
                next_obs_t = b["next_obs"]
                act_t = b["actions"]
                rew_t = b["rewards"]
                done_t = b["done"]
                oh_t = b["one_hot"]

                with tf.GradientTape(persistent=True) as g:
                    logits_cur = actor(obs_t, oh_t)
                    dist_cur = Categorical(logits=logits_cur)
                    entropy = dist_cur.entropy()

                    logits_next = actor(next_obs_t, oh_t)
                    dist_next = Categorical(logits=logits_next)
                    entropy_next = dist_next.entropy()

                    q1_all = critic1(obs_t, oh_t)
                    q2_all = critic2(obs_t, oh_t)
                    q1_vals = tf.gather(q1_all, act_t, axis=1, batch_dims=1)
                    q2_vals = tf.gather(q2_all, act_t, axis=1, batch_dims=1)

                    tq1 = target_critic1(next_obs_t, oh_t)
                    tq2 = target_critic2(next_obs_t, oh_t)
                    min_tq = dist_next.probs_parameter() * tf.minimum(tq1, tq2)
                    backup = tf.stop_gradient(
                        rew_t + gamma * (1 - done_t) * (tf.reduce_sum(min_tq, -1) - entropy_next)
                    )
                    q1_loss = 0.5 * tf.reduce_mean((backup - q1_vals) ** 2)
                    q2_loss = 0.5 * tf.reduce_mean((backup - q2_vals) ** 2)
                    value_loss = q1_loss + q2_loss

                    min_q = dist_cur.probs_parameter() * tf.stop_gradient(tf.minimum(q1_all, q2_all))
                    actor_loss = -tf.reduce_mean(entropy + tf.reduce_sum(min_q, -1))

                critic_vars = critic1.trainable_variables + critic2.trainable_variables
                c_grad = g.gradient(value_loss, critic_vars)
                a_grad = g.gradient(actor_loss, actor.trainable_variables)
                del g

                actor_optimizer.apply_gradients(zip(a_grad, actor.trainable_variables))
                critic_optimizer.apply_gradients(zip(c_grad, critic_vars))

                for v, tv in zip(critic1.trainable_variables, target_critic1.trainable_variables):
                    tv.assign(polyak * tv + (1 - polyak) * v)
                for v, tv in zip(critic2.trainable_variables, target_critic2.trainable_variables):
                    tv.assign(polyak * tv + (1 - polyak) * v)

        pbar.update(1)

        # Logging
        if (step + 1) % args.log_every == 0:
            epoch = (step + 1) // args.log_every
            logger.log_tabular("epoch", epoch)
            logger.log_tabular("total_env_steps", step + 1)
            logger.log_tabular("train/return", with_min_and_max=True)
            logger.log_tabular("train/ep_length", average_only=True)
            logger.log_tabular("walltime", time.time() - start_time)
            logger.dump_tabular()

    pbar.close()
    env.close()
    print("Single-scene training complete.")


if __name__ == "__main__":
    main()
