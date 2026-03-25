# Adding a New Algorithm to MariHA

This guide explains how to implement a new RL algorithm and run it against
the full MariHA benchmark — the same curriculum, CL metrics, and behavioral
metrics used by all built-in algorithms (SAC, DDQN, PPO, MuZero, …).

---

## What the benchmark provides

When your algorithm runs, the benchmark supplies:

| What | Where | Notes |
|---|---|---|
| Human-aligned episode curriculum | `ContinualLearningEnv` | Yields `EpisodeSpec` objects in subject order |
| Task identity | `info["task_one_hot"]` | One-hot vector; ignore if task-agnostic |
| Task boundary signal | `info["task_switch"]` (bool) | `True` on the first step of a new scene |
| Current scene ID | `info["scene_id"]` | e.g. `"w1l1s0"` |
| Structured logger | `EpochLogger` | TSV / TensorBoard / WandB |
| Canonical scene list | `scene_ids: list[str]` | Defines one-hot dimension and task index |
| Replay buffer implementations | `mariha.replay.buffers` | Optional — use if your algorithm is off-policy |

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

### 1. Create your algorithm file

```
mariha/rl/myalgo/
    __init__.py      ← empty
    agent.py         ← your implementation
```

### 2. Register it

Open `mariha/rl/__init__.py` and add two lines:

```python
from mariha.rl.myalgo.agent import MyAlgo
register("myalgo")(MyAlgo)
```

### 3. Run training

```bash
mariha-run --algorithm myalgo --subject sub-01 --seed 0
```

Pass your own hyperparameter flags if you defined them in `add_args`:

```bash
mariha-run --algorithm myalgo --subject sub-01 --my_lr 1e-4 --my_param 42
```

### 4. Run evaluation

```bash
mariha-evaluate --algorithm myalgo --subject sub-01 --run_prefix <timestamp_seed0>
```

This produces `experiments/sub-01/myalgo/<run_prefix>/eval_results.json` with
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
            json.dumps({"algorithm": "random", "n_actions": self.n_actions})
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
mariha-run --algorithm random --subject sub-01 --seed 0
mariha-evaluate --algorithm random --subject sub-01 --run_prefix <timestamp>
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
experiments/checkpoints/<algorithm>/<timestamp>_task<k>/
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
    "algorithm": "myalgo",
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

- [ ] Create `mariha/rl/myalgo/__init__.py` (empty)
- [ ] Create `mariha/rl/myalgo/agent.py` implementing `BenchmarkAgent`
- [ ] Implement `get_action`, `run`, `save_checkpoint`, `load_checkpoint`, `from_args`
- [ ] Add `register("myalgo")(MyAlgo)` to `mariha/rl/__init__.py`
- [ ] Test: `mariha-run --algorithm myalgo --subject sub-01 --seed 0`
- [ ] Evaluate: `mariha-evaluate --algorithm myalgo --subject sub-01 --run_prefix <ts>`
