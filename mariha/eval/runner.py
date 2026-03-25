"""Evaluation runner: execute a trained agent on a single scene.

``eval_on_scene`` runs the agent's deterministic (greedy) policy for
``n_episodes`` and returns the mean cumulative return plus per-episode
behavioral statistics (``EpisodeStats``).

The eval env is built via ``make_scene_env`` from the same wrapper pipeline
used during training, so observations and action spaces are identical.

The ``agent`` argument accepts any ``BenchmarkAgent`` implementation —
SAC, DDQN, PPO, MuZero, RandomAgent, or any custom algorithm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mariha.env.continual import make_scene_env
from mariha.env.scenario_gen import load_metadata
from mariha.env.base import SCENARIOS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_episode_stats(env) -> Optional[Dict]:
    """Traverse the wrapper chain to reach ``SceneEnv.episode_stats``."""
    inner = env
    while hasattr(inner, "_env"):
        inner = inner._env
    stats = getattr(inner, "episode_stats", None)
    return stats.to_dict() if stats is not None else None


# ---------------------------------------------------------------------------
# Core eval function
# ---------------------------------------------------------------------------


def eval_on_scene(
    agent,
    scene_id: str,
    eval_spec,
    scene_ids: List[str],
    n_episodes: int = 5,
    render_mode: Optional[str] = None,
    scenarios_dir: Path = SCENARIOS_DIR,
) -> Tuple[float, List[Dict]]:
    """Run the greedy policy on a single scene.

    Args:
        agent: A :class:`~mariha.benchmark.agent.BenchmarkAgent` instance.
        scene_id: Scene identifier, e.g. ``'w1l1s0'``.
        eval_spec: An :class:`~mariha.curriculum.episode.EpisodeSpec`
            describing the starting state file and step budget.
        scene_ids: Canonical ordered list of all scene IDs (for one-hot).
        n_episodes: Number of episodes to run.
        render_mode: Passed through to the env (``None`` = headless).
        scenarios_dir: Directory containing ``scenes_metadata.json``.

    Returns:
        A tuple ``(mean_return, stats_list)`` where ``stats_list`` is a list
        of dicts from ``EpisodeStats.to_dict()``, one per episode.
    """
    scene_meta = load_metadata(scenarios_dir)
    if scene_id not in scene_meta:
        raise ValueError(
            f"Scene '{scene_id}' not found in metadata. "
            "Run `mariha-generate-scenarios` first."
        )
    exit_point = scene_meta[scene_id]["exit_point"]

    env = make_scene_env(
        scene_id=scene_id,
        exit_point=exit_point,
        scene_ids=scene_ids,
        render_mode=render_mode,
        scenarios_dir=scenarios_dir,
    )

    returns: List[float] = []
    stats_list: List[Dict] = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset(episode_spec=eval_spec)
            one_hot = info["task_one_hot"]
            ep_return = 0.0
            done = False

            while not done:
                action = agent.get_action(obs, one_hot, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)
            ep_stats = _get_episode_stats(env)
            if ep_stats is not None:
                stats_list.append(ep_stats)

            logger.debug(
                "Scene %s  ep %d/%d  return=%.2f  cleared=%s",
                scene_id,
                ep + 1,
                n_episodes,
                ep_return,
                ep_stats.get("cleared", "?") if ep_stats else "?",
            )
    finally:
        env.close()

    mean_return = float(np.mean(returns)) if returns else 0.0
    return mean_return, stats_list


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_task_checkpoints(
    checkpoint_base: Path,
    run_prefix: str,
) -> Dict[int, Path]:
    """Discover per-task checkpoint directories for a run.

    Looks for directories matching ``{checkpoint_base}/{run_prefix}_task{k}``.

    Args:
        checkpoint_base: Directory containing all checkpoint dirs (e.g.
            ``experiments/checkpoints/ewc``).
        run_prefix: The timestamp + seed string, e.g. ``20240322_120000_seed0``.

    Returns:
        Dict mapping task index → checkpoint directory path.
    """
    result: Dict[int, Path] = {}
    for d in sorted(checkpoint_base.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(run_prefix + "_task"):
            continue
        suffix = name[len(run_prefix) + len("_task"):]
        try:
            task_idx = int(suffix)
        except ValueError:
            continue
        result[task_idx] = d
    return result
