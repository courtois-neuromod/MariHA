"""Continual learning training script.

Runs the SAC (or a CL baseline) on the full human-aligned curriculum for a
given subject.

Usage::

    python scripts/run_cl.py --subject sub-01 --cl_method ewc --seed 0

Or via the entry point installed by pyproject.toml::

    mariha-run-cl --subject sub-01 --cl_method ewc
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mariha.curriculum.sequences import HumanSequence
from mariha.env.continual import ContinualLearningEnv
from mariha.env.scenario_gen import load_metadata
from mariha.replay.buffers import BufferType
from mariha.rl import models
from mariha.rl.sac import SAC
from mariha.utils.config import build_parser
from mariha.utils.logging import EpochLogger
from mariha.utils.running import get_activation_from_str, get_readable_timestamp


def _build_cl_agent(cl_method: str | None, **sac_kwargs) -> SAC:
    """Instantiate the SAC or a CL-method subclass.

    Args:
        cl_method: Method name string or ``None`` for vanilla SAC.
        **sac_kwargs: Forwarded to the agent constructor.

    Returns:
        A ``SAC`` (or subclass) instance.
    """
    if cl_method is None:
        return SAC(**sac_kwargs)

    method = cl_method.lower()
    if method == "l2":
        from mariha.methods.l2 import L2_SAC
        return L2_SAC(**sac_kwargs)
    if method == "ewc":
        from mariha.methods.ewc import EWC_SAC
        return EWC_SAC(**sac_kwargs)
    if method == "mas":
        from mariha.methods.mas import MAS_SAC
        return MAS_SAC(**sac_kwargs)
    if method == "si":
        from mariha.methods.si import SI_SAC
        return SI_SAC(**sac_kwargs)
    if method == "owl":
        from mariha.methods.owl import OWL_SAC
        return OWL_SAC(**sac_kwargs)
    if method == "packnet":
        from mariha.methods.packnet import PackNet_SAC
        return PackNet_SAC(**sac_kwargs)
    if method == "agem":
        from mariha.methods.agem import AGEM_SAC
        return AGEM_SAC(**sac_kwargs)
    if method == "vcl":
        from mariha.methods.vcl import VCL_SAC
        return VCL_SAC(**sac_kwargs)
    if method in ("der", "der++"):
        from mariha.methods.der import DER_SAC
        return DER_SAC(**sac_kwargs)
    if method == "clonex":
        from mariha.methods.clonex import ClonEx_SAC
        return ClonEx_SAC(**sac_kwargs)
    if method == "multitask":
        from mariha.methods.multitask import MultiTask_SAC
        return MultiTask_SAC(**sac_kwargs)
    raise ValueError(f"Unknown cl_method: '{cl_method}'")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Curriculum
    # ------------------------------------------------------------------ #
    sequence = HumanSequence(
        subject_id=args.subject,
        require_existing_states=args.require_existing_states,
    )
    if len(sequence) == 0:
        print(f"ERROR: No episodes found for {args.subject}. Aborting.")
        sys.exit(1)

    # Full ordered list of scene IDs used by the benchmark (canonical ordering).
    # We derive it from the metadata so it's always consistent.
    from mariha.env.base import SCENARIOS_DIR
    scene_meta = load_metadata(SCENARIOS_DIR)
    scene_ids = sorted(scene_meta.keys())  # canonical alphabetical ordering

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    env = ContinualLearningEnv(
        sequence=sequence,
        scene_ids=scene_ids,
        render_mode=args.render_mode,
    )

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #
    timestamp = get_readable_timestamp()
    experiment_dir = Path(args.experiment_dir)
    run_dir = str(
        experiment_dir / args.subject / (args.cl_method or "sac") / f"{timestamp}_seed{args.seed}"
    )
    logger = EpochLogger(
        output_dir=run_dir,
        logger_output=args.logger_output,
        config=vars(args),
        group_id=f"{args.subject}_{args.cl_method or 'sac'}",
    )

    # ------------------------------------------------------------------ #
    # Policy kwargs
    # ------------------------------------------------------------------ #
    activation = get_activation_from_str(args.activation)
    policy_kwargs = dict(
        hidden_sizes=tuple(args.hidden_sizes),
        activation=activation,
        use_layer_norm=args.use_layer_norm,
        num_heads=args.num_heads,
        hide_task_id=args.hide_task_id,
    )

    # ------------------------------------------------------------------ #
    # Agent
    # ------------------------------------------------------------------ #
    sac_kwargs = dict(
        env=env,
        logger=logger,
        scene_ids=scene_ids,
        cl_method=args.cl_method,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        log_every=args.log_every,
        replay_size=args.replay_size,
        gamma=args.gamma,
        polyak=args.polyak,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_decay_rate=args.lr_decay_rate,
        alpha=args.alpha,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        n_updates=args.n_updates,
        save_freq_epochs=args.save_freq_epochs,
        reset_buffer_on_task_change=args.reset_buffer_on_task_change,
        buffer_type=BufferType(args.buffer_type),
        reset_optimizer_on_task_change=args.reset_optimizer_on_task_change,
        reset_actor_on_task_change=args.reset_actor_on_task_change,
        reset_critic_on_task_change=args.reset_critic_on_task_change,
        clipnorm=args.clipnorm,
        agent_policy_exploration=args.agent_policy_exploration,
        experiment_dir=experiment_dir,
        timestamp=timestamp,
    )

    agent = _build_cl_agent(args.cl_method, **sac_kwargs)

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    agent.run()
    env.close()


if __name__ == "__main__":
    main()
