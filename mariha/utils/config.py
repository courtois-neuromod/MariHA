"""Argument parsing and hyperparameter defaults for MariHA training scripts.

Provides ``build_parser()`` which returns an ``argparse.ArgumentParser``
pre-configured with all SAC and CL hyperparameters.  Individual scripts call
``parser.parse_args()`` and forward the resulting namespace to the SAC
constructor and logger.

.. deprecated::
    ``build_parser()`` is kept for backwards compatibility with
    ``run_single.py`` and any external code.  New scripts should use
    :func:`mariha.benchmark.config.build_benchmark_parser` for benchmark-level
    flags and :meth:`mariha.rl.sac.SAC.add_args` for SAC-specific flags.
"""

from __future__ import annotations

import argparse

from mariha.replay.buffers import BufferType
from mariha.utils.running import float_or_str, get_activation_from_str, sci2int, str2bool


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser shared by ``run_single.py`` and ``run_cl.py``.

    Composes the benchmark-level parser with all SAC-specific flags so that
    existing code calling ``build_parser()`` continues to work unchanged.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    p = argparse.ArgumentParser(
        description="MariHA SAC training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Curriculum ---
    p.add_argument("--subject", type=str, default="sub-01",
                   help="Subject ID for the HumanSequence curriculum.")
    p.add_argument("--scene_id", type=str, default=None,
                   help="(run_single only) Single scene ID to train on.")
    p.add_argument("--require_existing_states", type=str2bool, default=True,
                   help="Skip clips whose .state file is missing.")

    # --- SAC core ---
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--polyak", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=str, default=None,
                   choices=[None, "exponential", "linear"])
    p.add_argument("--lr_decay_rate", type=float, default=0.1)
    p.add_argument("--alpha", type=float_or_str, default="auto",
                   help="Entropy coefficient. Use 'auto' for automatic tuning.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--replay_size", type=sci2int, default=int(1e5))
    p.add_argument("--start_steps", type=sci2int, default=10_000)
    p.add_argument("--update_after", type=sci2int, default=5_000)
    p.add_argument("--update_every", type=int, default=50)
    p.add_argument("--n_updates", type=int, default=50)
    p.add_argument("--clipnorm", type=float, default=None)

    # --- Network architecture ---
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    p.add_argument("--activation", type=str, default="tanh",
                   choices=["tanh", "relu", "elu", "lrelu"])
    p.add_argument("--use_layer_norm", type=str2bool, default=False)
    p.add_argument("--num_heads", type=int, default=1,
                   help="Number of output heads (1 = shared, >1 = multi-head).")
    p.add_argument("--hide_task_id", type=str2bool, default=False)

    # --- Replay buffer ---
    p.add_argument("--buffer_type", type=str, default="fifo",
                   choices=[bt.value for bt in BufferType])

    # --- Task-change behaviour ---
    p.add_argument("--reset_buffer_on_task_change", type=str2bool, default=True)
    p.add_argument("--reset_optimizer_on_task_change", type=str2bool, default=False)
    p.add_argument("--reset_actor_on_task_change", type=str2bool, default=False)
    p.add_argument("--reset_critic_on_task_change", type=str2bool, default=False)
    p.add_argument("--agent_policy_exploration", type=str2bool, default=False)

    # --- CL method ---
    p.add_argument("--cl_method", type=str, default=None,
                   choices=[None, "l2", "ewc", "mas", "si", "owl", "packnet",
                             "agem", "vcl", "der", "clonex", "multitask"])

    # --- Logging ---
    p.add_argument("--log_every", type=sci2int, default=1000,
                   help="Steps between logging epochs.")
    p.add_argument("--save_freq_epochs", type=int, default=25)
    p.add_argument("--experiment_dir", type=str, default="experiments")
    p.add_argument("--logger_output", type=str, nargs="+",
                   default=["tsv", "tensorboard"],
                   help="Logging backends: tsv, tensorboard, wandb.")
    p.add_argument("--render_mode", type=str, default=None,
                   help="Render mode for MarioEnv (None or 'human').")

    return p
