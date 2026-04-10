# Adding a New Agent to MariHA

This guide explains how to implement a new RL agent and run it against
the full MariHA benchmark — the same curriculum, CL metrics, and behavioral
metrics used by all built-in agents (SAC, PPO, DQN, …).

There are two paths, and you should pick the one that matches your goals:

| Path                                          | Use it when                                                                                                          | Code size                  |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **A. `BaseAgent` mixin** (`mariha.rl.base`)   | You want to plug a standard episode-driven RL agent into the same training loop SAC/PPO/DQN already use, *and* you want every CL method to work on your agent for free. | ~100–150 lines             |
| **B. `BenchmarkAgent` from scratch**          | Your agent is exotic enough that the standard loop doesn't fit (e.g. an offline RL agent, an LLM-prompted policy, a planner with no gradient steps), or you simply want to copy `RandomAgent` and start hacking. | one self-contained file    |

This guide covers path **B** in detail (the canonical
`RandomAgent` template) and points to the right places in the codebase
for path **A**. Most contributors writing a new gradient-based RL
agent should pick path **A** — see the `Path A` section at the bottom
of this page.

---

## What the benchmark provides

When your agent runs, the benchmark supplies:

| What | Where | Notes |
|---|---|---|
| Human-aligned episode curriculum | `ContinualLearningEnv` | Yields `EpisodeSpec` objects in subject order |
| Task identity | `info["task_one_hot"]` | One-hot vector; ignore if task-agnostic |
| Task boundary signal | `info["task_switch"]` (bool) | `True` on the first step of a new scene |
| Current scene ID | `info["scene_id"]` | e.g. `"w1l1s0"` |
| Structured logger | `EpochLogger` | TSV / TensorBoard / WandB |
| Canonical scene list | `scene_ids: list[str]` | Defines one-hot dimension and task index |
| Replay buffer implementations | `mariha.replay.buffers` | Optional — use if your agent is off-policy |

The environment exposes a standard Gymnasium interface:
`reset() → (obs, info)` and `step(action) → (obs, reward, terminated, truncated, info)`.
Observations are stacked grayscale frames `(84, 84, 4)`.
Actions are discrete (`gymnasium.spaces.Discrete`).

---

## What you must implement

Subclass `BenchmarkAgent` from `mariha.benchmark.agent` and implement:

| Method | Required | Description |
|---|---|---|
| `get_action(obs, task_one_hot, deterministic)` | Yes | Return an int action |
| `run()` | Yes | Full training loop |
| `save_checkpoint(directory)` | Yes | Persist model to `directory/` |
| `load_checkpoint(directory)` | Yes | Restore from `directory/` |
| `on_task_start(task_idx, scene_id)` | No | Task boundary hook (default: no-op) |
| `on_task_end(task_idx)` | No | Task boundary hook (default: no-op) |
| `add_args(cls, parser)` | No | Add CLI flags (default: no-op) |
| `from_args(cls, args, env, logger, scene_ids)` | Yes | Construct agent from args |

---

## Step-by-step

### 1. Create your agent file

```
mariha/rl/myagent/
    __init__.py      ← empty
    agent.py         ← your implementation
```

### 2. Register it

Open `mariha/rl/__init__.py` and add two lines:

```python
from mariha.rl.myagent.agent import MyAgent
register("myagent")(MyAgent)
```

### 3. Run training

```bash
mariha-run-cl --agent myagent --subject sub-01 --seed 0
```

Pass your own hyperparameter flags if you defined them in `add_args`:

```bash
mariha-run-cl --agent myagent --subject sub-01 --my_lr 1e-4 --my_param 42
```

### 4. Run evaluation

```bash
mariha-evaluate --agent myagent --subject sub-01 --run_prefix <timestamp_seed0>
```

This produces `experiments/sub-01/myagent/<run_prefix>/eval_results.json` with
AP, BWT, forgetting, plasticity, and per-scene behavioral metrics.

---

## Minimal working example: `RandomAgent`

The following is a complete, runnable implementation.  It selects actions
uniformly at random and never learns — useful as a copy-paste template and
as a performance lower bound in comparisons.

```python
# mariha/rl/random/agent.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mariha.benchmark.agent import BenchmarkAgent
from mariha.utils.logging import EpochLogger


class RandomAgent(BenchmarkAgent):
    """Selects actions uniformly at random.  No learning occurs."""

    def __init__(self, env, logger: EpochLogger, scene_ids: list[str],
                 seed: int = 0, experiment_dir: Path = Path("experiments"),
                 timestamp: str = "") -> None:
        self.env = env
        self.logger = logger
        self.scene_ids = scene_ids
        self.n_actions = env.action_space.n
        self.rng = np.random.default_rng(seed)
        self.experiment_dir = experiment_dir
        self.timestamp = timestamp

    # --- Action selection ---

    def get_action(self, obs, task_one_hot, deterministic=False) -> int:
        return int(self.rng.integers(self.n_actions))

    # --- Training loop ---

    def run(self) -> None:
        obs, info = self.env.reset()
        current_scene_id = info.get("scene_id", "")
        current_task_idx = (
            self.scene_ids.index(current_scene_id)
            if current_scene_id in self.scene_ids else 0
        )
        one_hot_vec = info["task_one_hot"]
        self.on_task_start(current_task_idx, current_scene_id)

        episode_return, episode_len, episodes = 0.0, 0, 0

        while True:
            action = self.get_action(obs, one_hot_vec)
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_return += reward
            episode_len += 1

            if terminated or truncated:
                episodes += 1
                self.logger.log(f"Ep {episodes} | return={episode_return:.2f}")
                episode_return, episode_len = 0.0, 0

                if self.env.is_done:
                    break

                obs, info = self.env.reset()
                one_hot_vec = info["task_one_hot"]

                if info.get("task_switch", False):
                    self.on_task_end(current_task_idx)
                    self.save_checkpoint(self._checkpoint_dir(current_task_idx))
                    new_scene = info.get("scene_id", "")
                    current_task_idx = (
                        self.scene_ids.index(new_scene)
                        if new_scene in self.scene_ids else current_task_idx
                    )
                    current_scene_id = new_scene
                    self.on_task_start(current_task_idx, current_scene_id)

        self.on_task_end(current_task_idx)
        self.save_checkpoint(self._checkpoint_dir(current_task_idx))

    # --- Checkpointing ---

    def save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "agent.json").write_text(
            json.dumps({"agent": "random", "n_actions": self.n_actions})
        )

    def load_checkpoint(self, directory: Path) -> None:
        pass  # nothing to restore

    # --- Config ---

    @classmethod
    def add_args(cls, parser) -> None:
        pass  # no hyperparameters

    @classmethod
    def from_args(cls, args, env, logger, scene_ids) -> "RandomAgent":
        from mariha.utils.running import get_readable_timestamp
        return cls(
            env=env, logger=logger, scene_ids=scene_ids,
            seed=getattr(args, "seed", 0),
            experiment_dir=Path(getattr(args, "experiment_dir", "experiments")),
            timestamp=get_readable_timestamp(),
        )

    def _checkpoint_dir(self, task_idx: int) -> Path:
        return self.experiment_dir / "checkpoints" / "random" / f"{self.timestamp}_task{task_idx}"
```

Register it in `mariha/rl/__init__.py`:

```python
from mariha.rl.random.agent import RandomAgent
register("random")(RandomAgent)
```

Run it:

```bash
mariha-run-cl --agent random --subject sub-01 --seed 0
mariha-evaluate --agent random --subject sub-01 --run_prefix <timestamp>
```

---

## Key patterns to follow

### Task switches

Detect task boundaries via `info["task_switch"]` returned by
`ContinualLearningEnv.reset()`:

```python
obs, info = self.env.reset()
if info.get("task_switch", False):
    self.on_task_end(prev_task_idx)
    # ... update current_task_idx ...
    self.on_task_start(new_task_idx, info["scene_id"])
```

### Saving checkpoints

The benchmark's diagonal evaluation (`mariha-evaluate --eval_diagonal`) loads
one checkpoint per task.  Save at every task boundary at minimum:

```python
self.save_checkpoint(self._checkpoint_dir(task_idx))
```

The directory naming convention that `mariha-evaluate` expects is:
```
experiments/checkpoints/<agent>/<timestamp>_task<k>/
```

The files *inside* that directory are yours to choose.

### Using replay buffers

Off-policy algorithms can reuse the ready-made buffers:

```python
from mariha.replay.buffers import ReplayBuffer, BufferType

self.replay_buffer = ReplayBuffer(
    obs_shape=env.observation_space.shape,
    size=100_000,
    num_tasks=len(scene_ids),
)
# Store a transition:
self.replay_buffer.store(obs, action, reward, next_obs, done, task_one_hot)
# Sample a batch (returns dict of TF tensors):
batch = self.replay_buffer.sample_batch(batch_size=128)
```

### Task identity

The one-hot task vector is always available in `info["task_one_hot"]`.
You can concatenate it to your network input (as SAC does), use it to index
per-task parameters, or ignore it entirely for task-agnostic methods.

---

## Reading the results

`mariha-evaluate` writes `eval_results.json` with:

```json
{
  "metadata": {
    "subject": "sub-01",
    "agent": "myagent",
    "run_prefix": "...",
    "n_episodes": 5,
    "n_scenes": 42,
    "eval_diagonal": false
  },
  "cl_metrics": {
    "AP": 0.73,
    "AP_std": 0.12,
    "per_scene_final": {"w1l1s0": 0.81, ...}
  },
  "behavioral_summary": {
    "clear_rate": 0.61,
    "mean_x_traveled": 1847.3,
    "mean_score_gained": 420.0,
    "death_rate": 0.22
  },
  "per_scene_behavioral": { ... }
}
```

With `--eval_diagonal`, `cl_metrics` also includes `BWT`, `forgetting`, and
`plasticity`.

---

## Summary checklist

- [ ] Create `mariha/rl/myagent/__init__.py` (empty)
- [ ] Create `mariha/rl/myagent/agent.py` implementing `BenchmarkAgent`
- [ ] Implement `get_action`, `run`, `save_checkpoint`, `load_checkpoint`, `from_args`
- [ ] Add `register("myagent")(MyAgent)` to `mariha/rl/__init__.py`
- [ ] Test: `mariha-run-cl --agent myagent --subject sub-01 --seed 0`
- [ ] Evaluate: `mariha-evaluate --agent myagent --subject sub-01 --run_prefix <ts>`

---

## Path A: subclass `BaseAgent` and reuse the shared loop

The three built-in agents (SAC, PPO, DQN) all subclass
`mariha.rl.base.BaseAgent`, which is itself a `BenchmarkAgent` subclass
that owns the episode-driven training loop once and exposes a small
callback surface. If you go this route you do **not** write a `run()`
method — `BaseAgent.run()` is final and delegates to the shared
`TrainingLoopRunner`. You just implement the algorithm-specific pieces:

| Required callback                                          | What it does                                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `select_action(*, obs, one_hot, global_step, ...)`          | Return `(action, extras)` for one training step. `extras` is forwarded to `store_transition`. |
| `store_transition(*, obs, action, reward, ...)`             | Write the transition into your replay or rollout buffer.                                  |
| `should_update(global_step)`                                | Return `True` when `update_step` should fire. Per-step (SAC, DQN) or per-rollout (PPO).   |
| `update_step(*, global_step, current_task_idx)`             | Run one round of gradient updates.                                                        |
| `save_weights(directory)` / `load_weights(directory)`       | Per-task checkpoint I/O.                                                                  |
| `get_action(obs, one_hot, deterministic)`                   | The greedy evaluation action — same as the `BenchmarkAgent` contract.                     |

Optional callbacks have sensible defaults:
`on_task_change`, `handle_session_boundary`, `on_episode_end`,
`log_after_epoch`, `get_log_tabular_keys`, `on_burn_in_start`,
`on_burn_in_end`. Override only what you need.

If you also want every CL method to work on your agent, implement the
three agent-agnostic CL hooks:

| CL hook                                          | What it returns                                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `get_named_parameter_groups()`                   | `dict[str, list[tf.Variable]]` of logical parameter groups. Pick names from `actor`, `policy`, `q`, `critic1`, `critic2`, `value`, or invent your own. |
| `forward_for_importance(obs, one_hot)`           | One differentiable output tensor per group, used by EWC/MAS.                                             |
| `distill_targets(obs, one_hot)`                  | Targets for distillation methods. Return `{"actor_logits": ...}` for stochastic policies, `{"q_values": ...}` for value-based agents. |

The default group resolution `actor → policy → q` means that as long
as one of those keys exists in your `get_named_parameter_groups`
output, every regularizer (L2, EWC, MAS, SI), pruner (PackNet), and
projector (AGEM) will pick the right target without any per-method
configuration.

A 130-line A2C skeleton that opts into the full agent-agnostic CL
matrix looks like this:

```python
from mariha.benchmark.registry import register
from mariha.rl.base import BaseAgent

@register("a2c")
class A2C(BaseAgent):
    update_granularity = "per_rollout"  # informational

    def __init__(self, env, logger, scene_ids, *, agent_name="a2c",
                 lr=3e-4, gamma=0.99, rollout_length=128, **kwargs):
        super().__init__(env, logger, scene_ids,
                         agent_name=agent_name, **kwargs)
        # ... build network, rollout buffer, optimizer ...

    # --- Required BaseAgent callbacks ---

    def select_action(self, *, obs, one_hot, global_step, task_step,
                      current_task_idx):
        ...
        return action, {"value": value, "log_prob": log_prob}

    def store_transition(self, *, obs, action, reward, next_obs,
                         terminated, truncated, one_hot, scene_id,
                         info, extras):
        self.rollout.store(...)

    def should_update(self, global_step):
        return self.rollout.full

    def update_step(self, *, global_step, current_task_idx):
        ...one A2C update over the rollout...
        self.rollout.reset()

    def save_weights(self, directory):
        self.policy.save_weights(directory / "policy.h5")

    def load_weights(self, directory):
        self.policy.load_weights(directory / "policy.h5")

    def get_action(self, obs, task_one_hot, deterministic=False):
        ...

    # --- Optional: opt into agent-agnostic CL ---

    def get_named_parameter_groups(self):
        return {"policy": self.policy.trainable_variables}

    def forward_for_importance(self, obs, one_hot):
        return {"policy": self.policy(obs, one_hot)}

    def distill_targets(self, obs, one_hot):
        return {"actor_logits": self.policy(obs, one_hot)}

    # --- CLI plumbing ---

    @classmethod
    def add_args(cls, parser): ...
    @classmethod
    def from_args(cls, args, env, logger, scene_ids): ...
```

Run it the same way as any other agent — including with any CL method:

```bash
mariha-run-cl --agent a2c --subject sub-01 --seed 0
mariha-run-cl --agent a2c --cl_method ewc --subject sub-01 --seed 0
mariha-run-cl --agent a2c --cl_method packnet --subject sub-01 --seed 0
```

For a real example see `mariha/rl/dqn.py` (simplest of the three) or
`mariha/rl/ppo.py` (per-rollout pattern). For the agent-side
contract details look at the `BaseAgent` callback documentation in
`mariha/rl/base/agent_base.py`.
