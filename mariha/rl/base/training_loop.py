"""Episode-driven training loop shared by all :class:`BaseAgent` subclasses.

The loop is intentionally algorithm-agnostic: every step it consults the
agent via its callback surface (``select_action``, ``store_transition``,
``should_update``, ``update_step``) and lets the agent handle anything
specific to its algorithm.  The runner itself is concerned only with:

- Episode bookkeeping (return, length, action counts).
- Curriculum-level events (task switches, session boundaries, render
  checkpoints, final shutdown).
- Periodic logging and checkpointing cadence.

**Curriculum-level lifecycle**

The runner drives the agent through the human-aligned curriculum until
``ContinualLearningEnv.is_done`` flips to ``True``.  The lifecycle of
events the runner orchestrates, from outermost to innermost, looks like::

    run()
    │
    ├── env.reset()                                  ← initial reset
    ├── on_task_start(task=0, scene_id)              ← CL hook fires
    ├── on_task_change(task=0, scene_id)             ← agent-side reset
    │
    ├── for step in 0..∞:                            ← main loop
    │   │
    │   ├── select_action  → store_transition        ← per-step calls
    │   │
    │   ├── if episode_done:                         ← episode end (1/2)
    │   │       on_episode_end(...)
    │   │       maybe render_checkpoint()
    │   │
    │   ├── if should_update(step):
    │   │       update_step(step, task)              ← gradient step
    │   │
    │   ├── if (step+1) % log_every == 0:            ← epoch tick
    │   │       if epoch % save_freq_epochs == 0:
    │   │           save_weights(checkpoint_dir)     ← per-task ckpt
    │   │       log_after_epoch(epoch, step, ...)
    │   │
    │   └── if episode_done:                         ← episode end (2/2)
    │           if env.is_done: break                ← curriculum done
    │           obs, info = env.reset()
    │           if info["session_switch"]:
    │               handle_session_boundary(task)    ← per-scene flush
    │           if info["task_switch"]:
    │               on_task_end(old_task)            ← CL hook fires
    │               on_task_start(new_task, scene)   ← CL hook fires
    │               on_task_change(new_task, scene)  ← agent-side reset
    │
    └── on_task_end(final_task)                      ← final flush
        save_weights(checkpoint_dir)                 ← final ckpt

The five curriculum-level events fire in this order:

1. **Episode end** — the agent's episode bookkeeping (``on_episode_end``)
   runs first, before any policy update or logging.  The render
   checkpoint (live greedy episode in a window) also fires here.
2. **Session boundary** (``info["session_switch"] == True``) — fires
   when the curriculum advances to a new human play session within the
   same scene.  Used by per-scene buffer pools to flush pending
   transitions.
3. **Task switch** (``info["task_switch"] == True``) — fires when the
   curriculum advances to a new scene.  The CL method's ``on_task_end``
   hook runs *before* the agent's task-boundary reset (so importance
   estimators, PackNet pruning, and DER/ClonEx episodic memories can
   still read the just-finished task's replay buffer).  Then
   ``on_task_start`` and ``on_task_change`` fire with the new task
   index.
4. **Periodic checkpoint** — written every ``save_freq_epochs`` log
   epochs into ``experiments/checkpoints/{run_label}/{timestamp}_seed{seed}_task{k}/``.
   Multiple writes within the same task overwrite the same directory,
   so the final state of each task is what's available at evaluation
   time.
5. **Final shutdown** — once the env reports ``is_done``, the runner
   fires one last ``on_task_end`` and writes a final per-task
   checkpoint, then returns control to ``BaseAgent.run``.

**Per-step sequence**::

    select_action      →  (action, extras)
    env.step(action)   →  next_obs, reward, terminated, truncated, info
    store_transition   ←  runner forwards all of the above plus scene_id/extras
    obs ← next_obs
    episode_return, episode_len, action_counts updated

    if terminated or truncated:                # episode-end (part 1)
        on_episode_end(return, len, count)
        maybe render_checkpoint

    if should_update(global_step):
        update_step(global_step, current_task_idx)

    if (global_step + 1) % log_every == 0:
        if epoch % save_freq_epochs == 0:
            save_weights(standard_checkpoint_dir(...))
        log_after_epoch(epoch, global_step, action_counts)
        reset action_counts

    if terminated or truncated:                # episode-end (part 2)
        if env.is_done: break
        env.reset() or break
        if session_switch: handle_session_boundary(task_idx)
        if task_switch:
            on_task_end(old_idx)
            on_task_start(new_idx, new_scene_id)
            on_task_change(new_idx, new_scene_id)
            task_step ← 0

    global_step, task_step ← +1

The episode-end block is intentionally split in two: ``on_episode_end``
and the render hook fire *before* the policy update + periodic logging,
while ``env.reset`` and the session/task switch hooks fire *after*.
This means the policy update and the log dump always run with the *old*
task in scope (so action counts and any per-task metrics belong to the
task they came from), and the very last log epoch fires correctly when
the curriculum ends on a clean cadence boundary.

**Supported update modes**

The runner does not branch on the agent's ``update_granularity``.  Both
per-step (SAC, DQN) and per-rollout (PPO) agents use the exact same
loop; the difference is encoded inside ``should_update`` (per-step: "has
``update_every`` elapsed?"; per-rollout: "is my rollout buffer full?") and
``update_step`` (per-step: sample a batch and apply gradients;
per-rollout: bootstrap + GAE + ``n_epochs`` of minibatch updates).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from mariha.rl.base.agent_base import BaseAgent


class TrainingLoopRunner:
    """Executes the shared episode-driven training loop for a :class:`BaseAgent`.

    Instantiated from :meth:`BaseAgent.run`; not intended to be used
    directly by user code.  All algorithm-specific behaviour lives on the
    agent; the runner owns only the loop structure, episode bookkeeping,
    and curriculum-level event dispatch.
    """

    def __init__(self, agent: "BaseAgent") -> None:
        self.agent = agent
        self.env = agent.env
        self.logger = agent.logger

        # Runtime state — reset at the start of :meth:`run`.
        self._current_task_idx: int = 0
        self._current_scene_id: str = ""
        self._current_session: str = ""
        self._global_step: int = 0
        self._task_step: int = 0
        self._episodes: int = 0
        self._episode_return: float = 0.0
        self._episode_len: int = 0
        self._action_counts: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full curriculum until the environment reports ``is_done``."""
        self.agent.start_time = time.time()

        # --- initial reset ---
        try:
            obs, info = self.env.reset()
        except StopIteration:
            self.logger.log(
                "Curriculum is empty — nothing to train.", color="red"
            )
            return

        one_hot_vec = info["task_one_hot"]
        self._current_scene_id = info.get("scene_id", "")
        self._current_session = info.get("session", "")
        self._current_task_idx = (
            self.agent.scene_ids.index(self._current_scene_id)
            if self._current_scene_id in self.agent.scene_ids
            else 0
        )
        self._action_counts = {i: 0 for i in range(self.agent.act_dim)}
        self._global_step = 0
        self._task_step = 0
        self._episodes = 0
        self._episode_return = 0.0
        self._episode_len = 0

        # Fire lifecycle hooks for the first task.  on_task_change runs
        # after on_task_start so that agents see an already-announced task
        # when they perform any per-task resets (buffer reinit, etc.).
        self.agent.on_task_start(self._current_task_idx, self._current_scene_id)
        self.agent.on_task_change(self._current_task_idx, self._current_scene_id)

        self.logger.log(
            f"{self.agent.agent_name} training started.", color="green"
        )

        # --- main loop ---
        while True:
            # ---- action selection ----
            action, extras = self.agent.select_action(
                obs=obs,
                one_hot=one_hot_vec,
                global_step=self._global_step,
                task_step=self._task_step,
                current_task_idx=self._current_task_idx,
            )

            # ---- environment step ----
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # ---- transition storage ----
            self.agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                one_hot=one_hot_vec,
                scene_id=self._current_scene_id,
                info=info,
                extras=extras,
            )

            obs = next_obs
            self._episode_return += float(reward)
            self._episode_len += 1
            self._action_counts[action] = (
                self._action_counts.get(action, 0) + 1
            )

            # ---- end-of-episode bookkeeping ----
            # The episode-end block is split in two: this part (running
            # *before* the policy update + periodic logging) records the
            # episode that just finished and fires the render hook.  The
            # second half (env.reset + session/task switch handling) runs
            # *after* the update + log block so that the very last log
            # epoch fires correctly when the curriculum ends on a clean
            # cadence boundary, and so that update_step / log_after_epoch
            # always run with the *old* task index in scope.
            if done:
                self._episodes += 1
                self.agent.on_episode_end(
                    episode_return=self._episode_return,
                    episode_len=self._episode_len,
                    total_episodes=self._episodes,
                )
                self._episode_return = 0.0
                self._episode_len = 0

                # Render cadence: open a live window with a greedy
                # episode, so the user can see training progress.
                if (
                    self.agent.render_every > 0
                    and self._episodes % self.agent.render_every == 0
                ):
                    self.logger.log(
                        f"[render] episode {self._episodes} — "
                        "opening live window...",
                        color="cyan",
                    )
                    self.env.render_checkpoint(self.agent.get_action)

            # ---- policy update ----
            if self.agent.should_update(self._global_step):
                self.agent.update_step(
                    global_step=self._global_step,
                    current_task_idx=self._current_task_idx,
                )

            # ---- periodic logging + checkpointing ----
            if (self._global_step + 1) % self.agent.log_every == 0:
                epoch = (self._global_step + 1) // self.agent.log_every
                if epoch % self.agent.save_freq_epochs == 0:
                    self._save_periodic_checkpoint()
                self.agent.log_after_epoch(
                    epoch=epoch,
                    global_step=self._global_step,
                    action_counts=self._action_counts,
                )
                self._action_counts = {
                    i: 0 for i in range(self.agent.act_dim)
                }

            # ---- post-update episode termination + reset ----
            # This is the second half of the episode-end block.  It
            # runs after the update + log so that those operations see
            # the old task's state, and so that a final log epoch fires
            # before the curriculum-exhausted break.
            if done:
                if self.env.is_done:
                    break

                try:
                    obs, info = self.env.reset()
                except StopIteration:
                    break

                one_hot_vec = info["task_one_hot"]
                new_scene_id = info.get("scene_id", "")
                new_session = info.get("session", "")

                # Session boundary (for per-scene buffer flushes, etc.)
                if info.get("session_switch", False):
                    self.agent.handle_session_boundary(
                        self._current_task_idx
                    )

                # Task switch — fire lifecycle hooks in the canonical
                # order: finish old task, announce new task, reset agent.
                if info.get("task_switch", False):
                    self.agent.on_task_end(self._current_task_idx)
                    new_task_idx = (
                        self.agent.scene_ids.index(new_scene_id)
                        if new_scene_id in self.agent.scene_ids
                        else self._current_task_idx
                    )
                    self.agent.on_task_start(new_task_idx, new_scene_id)
                    self.agent.on_task_change(new_task_idx, new_scene_id)
                    self._current_task_idx = new_task_idx
                    self._current_scene_id = new_scene_id
                    self._task_step = 0

                self._current_session = new_session

            self._global_step += 1
            self._task_step += 1

        # --- training complete ---
        self.agent.on_task_end(self._current_task_idx)
        self._save_periodic_checkpoint()
        self.logger.log(
            f"{self.agent.agent_name} training complete — "
            f"{self._global_step} steps, {self._episodes} episodes.",
            color="green",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_periodic_checkpoint(self) -> None:
        """Write a per-task checkpoint under the standard directory layout."""
        from mariha.rl.base.checkpoint import standard_checkpoint_dir

        directory = standard_checkpoint_dir(
            self.agent.experiment_dir,
            self.agent.agent_name,
            self.agent.timestamp,
            self._current_task_idx,
        )
        directory.mkdir(parents=True, exist_ok=True)
        self.logger.log(
            f"Saving {self.agent.agent_name} checkpoint to {directory}",
            color="crimson",
        )
        self.agent.save_weights(directory)
