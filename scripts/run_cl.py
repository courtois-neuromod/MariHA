"""Continual learning training script.

Runs any registered agent on the full human-aligned curriculum for a
given subject.

Usage::

    mariha-run-cl --agent sac --subject sub-01 --seed 0
    mariha-run-cl --agent ewc --subject sub-01 --seed 0
    mariha-run-cl --agent ppo --subject sub-01 --seed 0

Run ``mariha-run-cl --agent <name> --help`` to see agent-specific flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # ------------------------------------------------------------------
    # Phase 1: parse benchmark-only flags so we know which agent to use
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_benchmark_parser

    bench_parser = build_benchmark_parser()
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
        description=f"MariHA CL benchmark — {bench_args.agent}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bench_parser],
        add_help=True,
    )
    agent_cls.add_args(full_parser)
    args = full_parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 4: build curriculum, environment, and logger
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_benchmark_context

    env, scene_ids, logger, sequence = build_benchmark_context(args)

    # ------------------------------------------------------------------
    # Phase 5: instantiate agent and run
    # ------------------------------------------------------------------
    agent = agent_cls.from_args(args, env=env, logger=logger, scene_ids=scene_ids)

    # Burn-in phase (pre-curriculum warm-up on a single scene).
    burn_in_steps = getattr(args, "burn_in_steps", 0)
    if burn_in_steps > 0:
        burn_in_scene = getattr(args, "burn_in_scene", "w1l1s0")
        burn_in_spec = next(
            (s for s in sequence if s.scene_id == burn_in_scene), None
        )
        if burn_in_spec is None:
            logger.log(
                f"WARNING: burn-in scene '{burn_in_scene}' not found in "
                "curriculum. Skipping burn-in.",
                color="yellow",
            )
        else:
            agent.burn_in(burn_in_spec, burn_in_steps)

    agent.run()
    env.close()


if __name__ == "__main__":
    main()
