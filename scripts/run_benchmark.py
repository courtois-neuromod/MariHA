"""Algorithm-agnostic MariHA benchmark training entry point.

Runs any registered algorithm on a human-aligned curriculum.

Usage::

    mariha-run --algorithm sac    --subject sub-01 --seed 0 --lr 1e-3
    mariha-run --algorithm ewc    --subject sub-01 --seed 0 --ewc_lambda 10000
    mariha-run --algorithm random --subject sub-01 --seed 0
    mariha-run --algorithm ddqn   --subject sub-01 --seed 0 --lr 1e-4

Run ``mariha-run --algorithm <name> --help`` to see the algorithm-specific
flags for any registered algorithm.

How it works
------------
1. Parse benchmark-only flags (``--algorithm``, ``--subject``, …).
2. Look up the algorithm class from the registry.
3. Let the algorithm add its own flags to the parser.
4. Re-parse all flags together.
5. Build curriculum, environment, and logger.
6. Instantiate the agent via ``agent_cls.from_args(...)`` and call ``run()``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # ------------------------------------------------------------------
    # Phase 1: parse benchmark-only flags so we know which algorithm to use
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_benchmark_parser
    bench_parser = build_benchmark_parser()
    bench_args, _ = bench_parser.parse_known_args()

    # ------------------------------------------------------------------
    # Phase 2: trigger registry population, look up algorithm class
    # ------------------------------------------------------------------
    import mariha.rl  # noqa: F401 — registers all built-in algorithms
    from mariha.benchmark.registry import get_agent_class, list_agents
    try:
        agent_cls = get_agent_class(bench_args.algorithm)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print(f"Registered algorithms: {list_agents()}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: build the full parser with algorithm-specific flags
    # ------------------------------------------------------------------
    full_parser = argparse.ArgumentParser(
        description=f"MariHA benchmark — {bench_args.algorithm}",
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
    env, scene_ids, logger = build_benchmark_context(args)

    # ------------------------------------------------------------------
    # Phase 5: instantiate agent and run
    # ------------------------------------------------------------------
    agent = agent_cls.from_args(args, env=env, logger=logger, scene_ids=scene_ids)
    agent.run()
    env.close()


if __name__ == "__main__":
    main()
