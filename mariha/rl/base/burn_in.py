"""Shared burn-in driver for :class:`BaseAgent` subclasses.

All three current agents (SAC, PPO, DQN) have nearly identical burn-in
structure:

1. Build a dedicated :class:`SceneEnv` for a single scene.  stable-retro
   only allows one emulator per process, so the caller must release the
   main training env before invoking this helper — ``run_burn_in``
   builds its own env and closes it in the ``finally`` block.
2. Repeatedly play the burn-in scene, collecting transitions and
   running gradient updates, until ``num_steps`` have been seen.
3. Reset the schedule via :meth:`BaseAgent.on_burn_in_end` (typically
   flushing the buffer and promoting ``post_burn_in_update_after`` into
   ``update_after``).

The helper leans on the agent's regular callback surface
(``select_action`` / ``store_transition`` / ``should_update`` /
``update_step``) plus the two burn-in bookends
(:meth:`BaseAgent.on_burn_in_start`, :meth:`BaseAgent.on_burn_in_end`).
No task-switch or session-boundary handling is performed inside the
burn-in loop — burn-in is a single contiguous session on a single scene.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


def run_burn_in(
    agent: "BaseAgent",
    burn_in_spec,
    num_steps: int,
) -> None:
    """Run ``num_steps`` of burn-in on a dedicated single-scene env.

    Args:
        agent: The :class:`BaseAgent` to train.  Its callbacks are used
            throughout; the agent's main :attr:`env` is *not* touched.
        burn_in_spec: An :class:`EpisodeSpec` pointing at the burn-in
            scene.  The agent is repeatedly placed at this spec until
            ``num_steps`` transitions have been collected.
        num_steps: Total number of environment steps to gather.

    The caller (e.g. ``scripts/run_cl.py``) is responsible for releasing
    the main training env before invoking this helper, and for
    rebuilding it afterwards — stable-retro only permits one live
    emulator per process.
    """
    # Local imports keep this module cheap to import (no env stack
    # load-time dependencies) and also avoid a cycle with agent_base.
    from mariha.env.base import SCENARIOS_DIR
    from mariha.env.continual import make_scene_env
    from mariha.env.scenario_gen import load_metadata

    scene_id = burn_in_spec.scene_id
    agent.logger.log(
        f"[burn-in] {agent.agent_name} burn-in on '{scene_id}' "
        f"for {num_steps} steps.",
        color="cyan",
    )

    scene_meta = load_metadata(SCENARIOS_DIR)
    exit_point = scene_meta[scene_id]["exit_point"]
    burn_env = make_scene_env(
        scene_id=scene_id,
        exit_point=exit_point,
        scene_ids=agent.scene_ids,
        render_mode=None,
    )

    task_idx = (
        agent.scene_ids.index(scene_id)
        if scene_id in agent.scene_ids
        else 0
    )
    agent.on_burn_in_start(task_idx)

    step = 0
    episodes = 0
    t_start = time.time()
    try:
        obs, info = burn_env.reset(episode_spec=burn_in_spec)
        one_hot_vec = info["task_one_hot"]

        while step < num_steps:
            action, extras = agent.select_action(
                obs=obs,
                one_hot=one_hot_vec,
                global_step=step,
                task_step=step,
                current_task_idx=task_idx,
            )
            next_obs, reward, terminated, truncated, step_info = (
                burn_env.step(action)
            )
            done = terminated or truncated

            agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                one_hot=one_hot_vec,
                scene_id=scene_id,
                info=step_info,
                extras=extras,
            )
            obs = next_obs

            if done:
                episodes += 1
                obs, info = burn_env.reset(episode_spec=burn_in_spec)
                one_hot_vec = info["task_one_hot"]

            if agent.should_update(step):
                agent.update_step(global_step=step, current_task_idx=task_idx)

            step += 1
    finally:
        burn_env.close()

    agent.on_burn_in_end()
    elapsed = time.time() - t_start
    agent.logger.log(
        f"[burn-in] Complete — {step} steps, {episodes} episodes "
        f"in {elapsed:.1f}s.",
        color="cyan",
    )
