"""Single-scene training script (development / debugging).

Trains any registered agent on a single Mario scene drawn from the human
curriculum.  Useful for verifying the environment, tuning hyperparameters,
or quick proof-of-concept runs before the full continual-learning experiment.

This script is a single-scene mirror of ``mariha-run-cl``: it follows the
same registry/plugin pattern, so registering a new agent in
``mariha/rl/__init__.py`` makes it available here automatically.

Usage::

    mariha-run-single --agent sac --subject sub-01 --scene_id w1l1s0
    mariha-run-single --agent ppo --subject sub-01 --scene_id w1l1s0
    mariha-run-single --agent dqn --subject sub-01 --scene_id w1l1s0

If ``--scene_id`` is omitted, the first scene in the subject's curriculum
is used.  ``--total_steps`` (default 200 000) is a soft env-step budget —
the in-flight episode is allowed to finish.

Pass ``--render_every N`` to open a live window every N training episodes
and watch the agent play one full greedy episode.  Combine it with
``--render_speed S`` to control playback speed (``1.0`` = native 60 fps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # ------------------------------------------------------------------
    # Phase 1: parse single-scene benchmark flags so we know which agent
    # to use
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_single_scene_parser

    bench_parser = build_single_scene_parser()
    bench_args, _ = bench_parser.parse_known_args()

    # ------------------------------------------------------------------
    # Phase 2: trigger registry population, look up agent class
    # ------------------------------------------------------------------
    import mariha.rl  # noqa: F401 — registers all built-in agents
    from mariha.benchmark.registry import get_agent_class, list_agents

    try:
        agent_cls = get_agent_class(bench_args.agent)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print(f"Registered agents: {list_agents()}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: build the full parser with agent-specific flags
    # ------------------------------------------------------------------
    full_parser = argparse.ArgumentParser(
        description=f"MariHA single-scene runner — {bench_args.agent}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bench_parser],
        add_help=True,
    )
    agent_cls.add_args(full_parser)
    args = full_parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 4: build single-scene env, scene_ids, and logger
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_single_scene_context

    env, scene_ids, logger, _ = build_single_scene_context(args)

    # ------------------------------------------------------------------
    # Phase 5: instantiate agent and run
    # ------------------------------------------------------------------
    agent = agent_cls.from_args(args, env=env, logger=logger, scene_ids=scene_ids)
    agent.run()
    env.close()


if __name__ == "__main__":
    main()
