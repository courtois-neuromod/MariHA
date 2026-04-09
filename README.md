# MariHA - Mario Human Alignment Benchmark (Status: WiP - untested)

> 🎵 Mario-hii Mario-huu Mario-hoo Mario ah haaaa! 🎵
>
> — An exhausted data manager.

**Built upon [COOM (Continual Doom)](https://github.com/TTomilin/COOM)** - This project is based on the [COOM](https://arxiv.org/abs/2303.13002) benchmark by Tristan Tomilin et al. (NeurIPS 2023). We extend their excellent continual reinforcement learning framework to create MariHA, a benchmark for evaluating human-AI alignment in game-based environments.


In MariHA, agents are trained on a human-derived curriculum — a sequence of short gameplay clips recorded from real players — and must progressively learn to clear scenes that the human subjects encountered during their sessions. The benchmark measures not only raw performance but also how well agents *retain* past skills and *transfer* knowledge to new scenes.


📄 **Paper**: [MariHA on OpenReview](https://openreview.net/forum?id=YAVB439L9X)

<p align="center">
  <img src="assets/gifs/mario_demo1.gif" alt="Mario Demo 1" style="vertical-align: top;"/>
  <img src="assets/gifs/mario_demo2.gif" alt="Mario Demo 2" style="vertical-align: top;"/>
</p>

---

## Overview

| Property | Value |
|----------|-------|
| Observation space | `(84, 84, 4)` grayscale frame-stack, normalised to `[0, 1]` |
| Action space | Discrete (9 actions) |
| Scenes | 313 unique scenes across 32 levels |
| Subjects | 5 (sub-01, 02, 03, 05, 06) |
| Task structure | Sequential curriculum derived from human play sessions |
| Reward | Δ X-position per step |
| Termination | Scene cleared (exit reached), death, or frame budget exhausted |

---

## Setup

```bash
bash setup.sh
source env/bin/activate
```

`setup.sh` creates a Python 3.9+ virtual environment, installs all dependencies, and generates the per-scene scenario files. Subject data (`.state` files) is stored in `data/mario.scenes` via git-annex:

```bash
# Pull data for all subjects
cd data/mario.scenes && git annex get sub-*/
```

---

## Quickstart

### Train (full CL curriculum)

```bash
mariha-run-cl --algorithm sac --subject sub-01 --seed 0
mariha-run-cl --algorithm ewc --subject sub-01 --seed 0
mariha-run-cl --algorithm ppo --subject sub-01 --seed 0
mariha-run-cl --algorithm dqn --subject sub-01 --seed 0
```

Pass `--render_every N` to open a live window every N episodes and watch the agent play in real time:

```bash
mariha-run-cl --algorithm ewc --subject sub-01 --seed 0 --render_every 100
```

### Train (single scene, for debugging)

```bash
mariha-run-single --scene_id w1l1s0 --seed 0
```

Pass `--render_every N` to open a live window every N episodes and watch the agent play in real time:

```bash
mariha-run-single --scene_id w1l1s0 --seed 0 --render_every 10
```

### Evaluate

```bash
mariha-evaluate \
  --subject sub-01 \
  --algorithm ewc \
  --run_prefix <timestamp_seed0> \
  --n_episodes 5 \
  --eval_diagonal          # adds BWT / forgetting metrics
```

Outputs `experiments/sub-01/ewc/<run_prefix>/eval_results.json`.

---

## Algorithms

| `--algorithm` | Class | Description |
|---------------|-------|-------------|
| `sac` | `SAC` | Vanilla SAC — fine-tune sequentially |
| `ppo` | `PPO` | Proximal Policy Optimization (on-policy) |
| `dqn` | `DQN` | Deep Q-Network (epsilon-greedy) |
| `ddqn` | `DQN` | Double DQN (alias for `dqn` with `--double_dqn=True`) |
| `random` | `RandomAgent` | Random action baseline |
| `l2` | `L2_SAC` | L2 regularisation (uniform importance) |
| `ewc` | `EWC_SAC` | Elastic Weight Consolidation (Fisher diagonal) |
| `mas` | `MAS_SAC` | Memory Aware Synapses (output sensitivity) |
| `si` | `SI_SAC` | Synaptic Intelligence (online surrogate) |
| `owl` | `OWL_SAC` | EWC + UCB1 bandit task weighting |
| `packnet` | `PackNet_SAC` | Iterative weight pruning and freezing |
| `agem` | `AGEM_SAC` | Averaged Gradient Episodic Memory |
| `vcl` | `VCL_SAC` | Variational Continual Learning (Bayesian weights) |
| `der` | `DER_SAC` | Dark Experience Replay++ (actor distillation) |
| `clonex` | `ClonEx_SAC` | ClonEx (actor + critic distillation) |
| `multitask` | `MultiTask_SAC` | Joint training upper bound (shared replay) |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **AP** | Average Performance across all scenes with the final model |
| **BWT** | Backward Transfer — how much final performance differs from peak performance per task (negative = forgetting) |
| **Forgetting** | `max(0, −BWT)` |
| **Plasticity** | Performance on the last task only |
| **clear_rate** | Fraction of episodes where Mario reached the exit |
| **mean_x_traveled** | Mean X-distance per episode |
| **mean_score_gained** | Mean score delta per episode |
| **death_rate** | Fraction of episodes ending in death |

---

## Project Structure

```
MariHA/
├── mariha/
│   ├── curriculum/    # EpisodeSpec, HumanSequence loader
│   ├── env/           # MarioEnv, SceneEnv, ContinualLearningEnv, wrappers
│   ├── replay/        # FIFO, Reservoir, PER, EpisodicMemory, Rollout buffers
│   ├── rl/            # SAC, PPO, DQN training loops + network architectures
│   ├── methods/       # 12 CL baseline implementations (SAC-based)
│   └── eval/          # CL metrics, eval runner
├── scripts/
│   ├── run_cl.py      # Full CL training (mariha-run-cl)
│   ├── run_single.py  # Single-scene training (mariha-run-single)
│   └── evaluate.py    # Evaluation (mariha-evaluate)
├── data/
│   ├── mario/         # Game integration + stimuli (bundled)
│   └── mario.scenes/  # Human gameplay data (git-annex)
├── setup.sh
└── pyproject.toml
```

---

## Key Design Choices

**Human-aligned curriculum.** Each episode uses the exact emulator state (`.state`) from a human player's clip, placing the agent at the same starting position with the same game state. This grounds the benchmark in real human play rather than arbitrary level resets.

**Two-input architecture.** The actor and critic take the pixel observation `(84, 84, 4)` and the task one-hot vector as *separate* inputs. The CNN processes pixels; the task ID conditions the dense trunk. This follows the [COOM](https://arxiv.org/abs/2303.13002) convention and enables multi-head outputs for task-specific policies.

**Episode-driven loop.** Training consumes one `EpisodeSpec` per episode from the curriculum until exhausted, rather than a fixed step budget. This preserves the human-aligned timing of each clip.

**Same scenes for eval.** Evaluation uses the first clip per scene from the training curriculum as the canonical eval episode, enabling direct comparison of training-time performance vs. final-checkpoint performance.

---

## Inspired by and built upon COOM

From COOM, MariHA inherits:

- The discrete-action SAC implementation and two-input `(obs, task_one_hot)` architecture
- The `Regularization_SAC` base class and the `l2`, `ewc`, `mas`, `agem`, `packnet`, `vcl` baselines
- The `fifo` / `reservoir` / `priority` / `per` replay buffer stack

MariHA extends COOM with:

- **Human-aligned curriculum**: episodes are derived from real human gameplay recordings (6 subjects, 313 scenes, ~5 k clips), rather than procedurally generated levels
- **NES Super Mario Bros environment**: `stable-retro` integration replacing the ViZDoom stack
- **Additional CL baselines**: `si`, `owl`, `der`, `clonex`, `multitask`
- **Evaluation suite**: CL performance matrix (AP, BWT, forgetting, plasticity) + behavioral metrics (`clear_rate`, `mean_x_traveled`, `death_rate`) computed from the same scenes used during training

```bibtex
@inproceedings{tomilin2023coom,
  title     = {{COOM}: A Game Benchmark for Continual Reinforcement Learning},
  author    = {Tomilin, Tristan and Fang, Meng and Zhang, Yudi and Pechenizkiy, Mykola},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.13002},
}
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.13+ (tested on 2.21; `tf_keras` is installed automatically for Keras 3 compatibility)
- stable-retro 0.9.2+
- The `mario.scenes` dataset (git-annex)

---

## Citation

If you use MariHA in your research, please cite:

```bibtex
@misc{mariha2026,
  title   = {MariHA: A Continual Reinforcement Learning Benchmark for Human-AI Alignment on Super Mario Bros},
  year    = {2026},
}
```
