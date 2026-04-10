# Adding a New Continual Learning Method

This guide walks through writing a new `CLMethod` for MariHA. Because
the CL layer is composition-based — `agent.cl_method = MyMethod(...)`,
not `class MyMethod(SAC): ...` — your new method will work on every
registered RL agent (SAC, PPO, DQN, and any future addition) as soon
as you register it.

The two examples in this guide are:

1. **L2** — the minimal template. About 30 lines of code, no data
   needed. Inherits from `ParameterRegularizer` and overrides one
   method.
2. **EWC** — the Jacobian-based template. About 50 lines on top of
   `ParameterRegularizer`, calls `agent.forward_for_importance` to
   compute Fisher diagonals.

---

## What MariHA gives you

When your method is composed onto an agent, the agent invokes your
hooks at fixed points in its training loop:

| Hook                                              | When                                                                                                                          |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `on_task_start(agent, task_idx)`                  | After the agent has finished its task-boundary resets. Eager Python — numpy is safe.                                          |
| `on_task_end(agent, task_idx)`                    | Before the agent resets state. The replay buffer still holds the just-finished task's data — the right place for snapshotting and importance estimation. |
| `compute_loss_penalty(agent, *, task_idx)`        | Inside the agent's `GradientTape`, every gradient step. Return a scalar `tf.Tensor` that gets added to the agent's loss. |
| `adjust_gradients(agent, grads_by_group, ...)`    | After the agent has computed gradients but before the optimizer step. Return a (possibly modified) `grads_by_group` dict. |
| `after_gradient_step(agent, grads_by_group, ...)` | After the optimizer applied the (adjusted) gradients. Runs inside the agent's `@tf.function` trace.                            |
| `get_episodic_batch(agent, *, task_idx)`          | Before each gradient step; the returned batch is forwarded verbatim to `adjust_gradients`. Return `None` to skip.              |

All hooks are no-ops by default; override only what you need.

The agent itself exposes three contracts that your method talks to:

| Agent contract                                  | Returns                                                                                            |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `agent.get_named_parameter_groups()`            | `dict[str, list[tf.Variable]]` — one entry per logical parameter group.                            |
| `agent.forward_for_importance(obs, one_hot)`    | `dict[str, tf.Tensor]` of differentiable per-group outputs (used for Jacobian-based importance).   |
| `agent.distill_targets(obs, one_hot)`           | `dict[str, tf.Tensor]` of distillation targets — keys depend on the agent (`actor_logits`, `q_values`, ...). |

For SAC the groups are `{"actor", "critic1", "critic2"}`; for PPO they
are `{"policy", "value"}`; for DQN there is a single `{"q"}` group.
The default group resolution `actor → policy → q` means a regularizer
written without an explicit group list automatically picks the right
target on each agent.

---

## Step-by-step

### 1. Pick your base class

| Base class                                                       | When to use                                                                              |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `mariha.methods.base.CLMethod`                                   | Bespoke methods that don't fit either of the families below (PackNet, AGEM, MultiTask). |
| `mariha.methods.regularizer_base.ParameterRegularizer`           | Anchor-style methods that compute per-parameter importance and add a quadratic penalty.  |
| `mariha.methods.distillation_base.DistillationMethod`            | Methods that snapshot past-task targets in an episodic memory and add a distillation gradient. |

The two intermediate bases handle the bookkeeping (`θ*` snapshots,
importance accumulation, episodic-memory storage, gradient injection)
so you only have to implement the math that distinguishes your method.

### 2. Subclass and implement only the math

Minimal `ParameterRegularizer` subclass — this *is* the entirety of
`mariha/methods/l2.py`:

```python
from mariha.benchmark.cl_registry import register_cl
from mariha.methods.regularizer_base import ParameterRegularizer

@register_cl("l2")
class L2Regularizer(ParameterRegularizer):
    """Uniform-importance L2 anchor: λ · Σ (θ − θ*)²."""

    name = "l2"
    _needs_data = False  # no replay sampling needed

    def _compute_importance(self, agent, batch):
        return {
            gn: [tf.ones_like(w) for w in self._reg_weights[gn]]
            for gn in self._regularize_groups
        }
```

That's it. The base class handles:

- Snapshotting `θ*` at every task boundary
- Accumulating per-parameter importance across tasks
- Computing the quadratic penalty `λ · Σ_g Σ_k w[g][k] · (θ[g][k] − θ*[g][k])²`
- Lazy initialisation of the storage `tf.Variable`s on first use
- Default group resolution (`actor → policy → q`)
- CLI flag wiring (`--cl_reg_coef`, `--regularize_groups`,
  `--importance_batches`, `--importance_batch_size`)

EWC is the data-driven counterpart: it computes the Fisher diagonal
from Jacobians of the per-group output. The method body is:

```python
@register_cl("ewc")
class EWC(ParameterRegularizer):
    name = "ewc"

    def _compute_importance(self, agent, batch):
        groups = agent.get_named_parameter_groups()
        with tf.GradientTape(persistent=True) as tape:
            outputs = agent.forward_for_importance(
                batch["obs"], batch["one_hot"]
            )
            summed = {
                gn: tf.reduce_sum(outputs[gn], axis=-1)
                for gn in self._regularize_groups
            }

        importance = {}
        for gn in self._regularize_groups:
            gs = tape.jacobian(summed[gn], groups[gn])
            per_param = []
            for var, g in zip(groups[gn], gs):
                if g is None:
                    per_param.append(tf.zeros_like(var))
                    continue
                fisher = tf.reduce_mean(g ** 2, axis=0)
                fisher = tf.clip_by_value(fisher, 1e-5, np.inf)
                per_param.append(fisher)
            importance[gn] = per_param
        del tape
        return importance
```

`agent.forward_for_importance(obs, one_hot)` returns a dict like
`{"actor": logits}` (SAC), `{"policy": logits}` (PPO), or
`{"q": q_values}` (DQN). The same code works on all three because
EWC iterates `self._regularize_groups`, which the base class resolved
to whichever of `actor`/`policy`/`q` is present on the host agent.

### 3. Register your method

Decorate the class with `@register_cl("name")` and import it from
`mariha/methods/__init__.py` so the decorator runs at import time:

```python
# mariha/methods/__init__.py
from mariha.methods.mymethod import MyMethod  # triggers @register_cl
```

That single import is all the wiring needed — `--cl_method mymethod`
becomes selectable on the CLI immediately.

### 4. Add CLI hyperparameters (optional)

If your method has hyperparameters beyond the ones the base class
already exposes, override `add_args` and `from_args`:

```python
@classmethod
def add_args(cls, parser):
    super().add_args(parser)              # inherit base flags
    parser.add_argument(
        "--my_extra_param", type=float, default=0.5,
        help="Custom hyperparameter for MyMethod.",
    )

@classmethod
def from_args(cls, args, agent):
    base = super().from_args(args, agent)  # for ParameterRegularizer
    base.my_extra_param = getattr(args, "my_extra_param", 0.5)
    return base
```

For non-regularizer methods, write `from_args` from scratch — see
`mariha/methods/agem.py` or `mariha/methods/packnet.py` for examples.

### 5. Run it

```bash
mariha-run-cl --agent sac --cl_method mymethod --subject sub-01 --seed 0
mariha-run-cl --agent ppo --cl_method mymethod --subject sub-01 --seed 0
mariha-run-cl --agent dqn --cl_method mymethod --subject sub-01 --seed 0
```

Same method, three agents, no agent-side changes.

---

## Hook ordering reference

For one task, the lifecycle is:

```
on_task_start(agent, idx)        ← CL hook fires here
  ├── (training loop body)
  │     for each gradient step:
  │       loss += compute_loss_penalty(agent, task_idx=idx)
  │       grads = ...compute gradients...
  │       grads = adjust_gradients(agent, grads, task_idx=idx,
  │                                episodic_batch=get_episodic_batch(...))
  │       optimizer.apply(grads)
  │       after_gradient_step(agent, grads, task_idx=idx)
  │
on_task_end(agent, idx)          ← CL hook fires here, BEFORE the agent's reset
                                    (replay buffer still holds task data)
on_task_change(new_idx, ...)      ← agent-side: buffer/optimizer/network resets
```

`on_task_end` runs *before* the agent's task-boundary reset by design:
that is the only place where importance estimators (EWC/MAS), pruners
(PackNet) and episodic-memory snapshots (DER, ClonEx) can read the
just-finished task's replay buffer.

Hooks that run inside `@tf.function` (`compute_loss_penalty`,
`adjust_gradients`, `after_gradient_step`) must use TensorFlow ops
only — no `.numpy()`, no Python `if` on tensor values, no calls into
Python-side state mutation. Use `tf.cond` for branching.

Hooks that run in eager Python (`on_task_start`, `on_task_end`,
`get_episodic_batch`) can do whatever you want.

---

## Where to look in the existing methods

| Pattern you want                                          | Read this file                                  |
| --------------------------------------------------------- | ----------------------------------------------- |
| Minimal regularizer (no data)                             | `mariha/methods/l2.py`                          |
| Jacobian-based importance over per-group outputs          | `mariha/methods/ewc.py` and `mas.py`            |
| Online importance accumulation (`after_gradient_step`)    | `mariha/methods/si.py`                          |
| Bespoke ABC subclass with mask state                      | `mariha/methods/packnet.py`                     |
| Gradient projection via reference gradients               | `mariha/methods/agem.py`                        |
| Episodic memory + distillation gradient injection         | `mariha/methods/distillation_base.py` (`der.py`, `clonex.py` are thin subclasses) |
| Forcing an agent flag without per-step state              | `mariha/methods/multitask.py`                   |

For the agent-side contract (what your method receives) start with the
`# CL composition` section of `mariha/rl/base/agent_base.py` and the
three hook implementations on each concrete agent: `get_named_parameter_groups`,
`forward_for_importance`, `distill_targets`.
