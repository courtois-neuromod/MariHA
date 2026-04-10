"""Continual learning training script.

Runs any registered agent on the full human-aligned curriculum for a
given subject.  An optional ``--cl_method`` flag composes any registered
continual-learning method on top of the agent.

Usage::

    # Vanilla agents (no CL strategy)
    mariha-run-cl --agent sac --subject sub-01 --seed 0
    mariha-run-cl --agent ppo --subject sub-01 --seed 0
    mariha-run-cl --agent dqn --subject sub-01 --seed 0

    # Compose any CL method on any agent
    mariha-run-cl --agent sac --cl_method ewc --subject sub-01 --seed 0
    mariha-run-cl --agent dqn --cl_method der --subject sub-01 --seed 0

Run ``mariha-run-cl --agent <name> --cl_method <name> --help`` to see
the union of agent-specific and CL-method-specific flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # ------------------------------------------------------------------
    # Phase 1: parse benchmark-only flags so we know which agent + CL
    # method to use
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_benchmark_parser

    bench_parser = build_benchmark_parser()
    bench_args, _ = bench_parser.parse_known_args()

    # ------------------------------------------------------------------
    # Phase 2: trigger registry population, look up agent + CL classes
    # ------------------------------------------------------------------
    import mariha.rl  # noqa: F401 — registers all built-in agents
    import mariha.methods  # noqa: F401 — registers all built-in CL methods
    from mariha.benchmark.cl_registry import get_cl_class, list_cl_methods
    from mariha.benchmark.registry import get_agent_class, list_agents

    try:
        agent_cls = get_agent_class(bench_args.agent)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print(f"Registered agents: {list_agents()}")
        sys.exit(1)

    cl_cls = None
    if bench_args.cl_method:
        try:
            cl_cls = get_cl_class(bench_args.cl_method)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            print(f"Registered CL methods: {list_cl_methods()}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: build the full parser with agent- and CL-method-specific
    # flags
    # ------------------------------------------------------------------
    title = bench_args.agent + (
        f" + {bench_args.cl_method}" if bench_args.cl_method else ""
    )
    full_parser = argparse.ArgumentParser(
        description=f"MariHA CL benchmark — {title}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bench_parser],
        add_help=True,
    )
    agent_cls.add_args(full_parser)
    if cl_cls is not None:
        cl_cls.add_args(full_parser)
    args = full_parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 4: build curriculum, environment, and logger
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_benchmark_context

    env, scene_ids, logger, sequence = build_benchmark_context(args)

    # ------------------------------------------------------------------
    # Phase 5: instantiate agent + CL method and run
    # ------------------------------------------------------------------
    agent = agent_cls.from_args(args, env=env, logger=logger, scene_ids=scene_ids)

    if cl_cls is not None:
        agent.cl_method = cl_cls.from_args(args, agent)
        # Tag the agent name so checkpoint dirs are namespaced by CL method,
        # mirroring the composite ``run_label`` used for log/output paths in
        # ``build_benchmark_context``.  This must happen before ``agent.run()``
        # since the runner reads ``agent.agent_name`` when building the
        # checkpoint path.
        agent.agent_name = f"{agent.agent_name}_{bench_args.cl_method}"
        logger.log(
            f"[run-cl] Composing CL method: {bench_args.cl_method}",
            color="cyan",
        )

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
            # stable-retro allows only one emulator per process, so
            # release the main env's inner scene env while burn-in
            # builds its own, then rebuild afterward.
            env.release_emulator()
            try:
                agent.burn_in(burn_in_spec, burn_in_steps)
            finally:
                env.reacquire_emulator()

    # The progress context manager owns the terminal display lifecycle.
    # ``env.close()`` is kept inside the ``with`` block so the final
    # episode's ``on_episode_end`` event can render before the display
    # (``LiveProgress.stop()``) tears down.
    with logger.progress:
        try:
            agent.run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
