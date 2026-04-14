"""Single-scene training script (development / debugging).

Trains any registered agent on a single Mario scene drawn from the human
curriculum.  Useful for verifying the environment, tuning hyperparameters,
or quick proof-of-concept runs before the full continual-learning experiment.

This script is a single-scene mirror of ``mariha-run-cl``: it follows the
same registry/plugin pattern, so registering a new agent in
``mariha/rl/__init__.py`` makes it available here automatically.  An
optional ``--cl_method`` flag composes any registered CL method on top
of the agent.

Usage::

    mariha-run-single --agent sac --subject sub-01 --scene_id w1l1s0
    mariha-run-single --agent ppo --subject sub-01 --scene_id w1l1s0
    mariha-run-single --agent dqn --cl_method ewc --scene_id w1l1s0

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
    # Phase 1: parse single-scene benchmark flags so we know which
    # agent + CL method to use
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_single_scene_parser

    bench_parser = build_single_scene_parser()
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
        description=f"MariHA single-scene runner — {title}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bench_parser],
        add_help=True,
    )
    agent_cls.add_args(full_parser)
    if cl_cls is not None:
        cl_cls.add_args(full_parser)
    args = full_parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 4: build single-scene env, scene_ids, and logger
    # ------------------------------------------------------------------
    from mariha.benchmark.config import build_single_scene_context

    env, run_ids, logger, _ = build_single_scene_context(args)

    # ------------------------------------------------------------------
    # Phase 5: instantiate agent + CL method and run
    # ------------------------------------------------------------------
    agent = agent_cls.from_args(args, env=env, logger=logger, run_ids=run_ids)

    if cl_cls is not None:
        agent.cl_method = cl_cls.from_args(args, agent)
        # Tag the agent name so checkpoint dirs are namespaced by CL method,
        # mirroring the composite ``run_label`` used for log/output paths in
        # ``build_single_scene_context``.  Must happen before ``agent.run()``.
        agent.agent_name = f"{agent.agent_name}_{bench_args.cl_method}"
        logger.log(
            f"[run-single] Composing CL method: {bench_args.cl_method}",
            color="cyan",
        )

    # The progress context manager owns the terminal display lifecycle.
    # ``env.close()`` is kept inside the ``with`` block so the final
    # episode's ``on_episode_end`` event can render before the display
    # tears down.
    with logger.progress:
        try:
            agent.run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
