"""Drift guard for the hand-maintained architecture specs.

``scripts/architecture_specs.py`` declares each agent's network topology by
hand (the agent-level wiring cannot be reliably auto-derived from code) and
powers the PNG panels under ``assets/model_architectures/``. These tests
introspect the *real* Keras models in ``mariha/rl/models.py`` and the agent
classes, and fail loudly if a spec no longer matches — so the committed
diagrams cannot silently become wrong.

What is checked:
    * layer-level structure (backbone shape, head output widths) — the kind
      of drift that happens silently when ``models.py`` is tweaked;
    * agent-level structure (which networks each agent builds) — via the
      agent ``__init__`` source, to catch a critic/target being added or
      removed.
"""

import inspect

import gymnasium as gym
import numpy as np
import pytest

from scripts.architecture_specs import get_spec

N_ACTIONS = 9
NUM_TASKS = 5
STATE = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84, 4), dtype=np.float32)
ACTION = gym.spaces.Discrete(N_ACTIONS)


@pytest.fixture(scope="module")
def models():
    """The real network module — imported lazily so TF only loads if tests run."""
    from mariha.rl import models as m
    return m


# --- Layer-level structure --------------------------------------------------


def test_backbone_is_four_conv_then_dense512(models):
    """The spec draws 4 conv blocks + a 512-d dense projection per network."""
    backbone = models.BaseCNN(STATE.shape)
    assert len(backbone.convs) == 4
    assert backbone.output_dim == 512


def test_ppo_head_widths_match_spec(models):
    ac = models.PPOActorCritic(STATE, ACTION, NUM_TASKS)
    assert ac.actor_head.units == N_ACTIONS
    assert ac.critic_head.units == 1

    ppo = get_spec("ppo", N_ACTIONS)
    assert len(ppo.networks) == 1, "PPO shares one actor-critic network"
    heads = {h.id: h.units for h in ppo.networks[0].heads}
    assert heads == {"actor": ac.actor_head.units, "critic": ac.critic_head.units}


def test_sac_head_widths_match_spec(models):
    actor = models.MlpActor(STATE, ACTION, NUM_TASKS)
    critic = models.MlpCritic(STATE, ACTION, NUM_TASKS)
    # head_mu / head are Sequential([Input, Dense(...)]); the Dense is last.
    assert actor.head_mu.layers[-1].units == N_ACTIONS
    assert critic.head.layers[-1].units == N_ACTIONS

    sac = get_spec("sac", N_ACTIONS)
    for net in sac.networks:
        for head in net.heads:
            assert head.units == N_ACTIONS, f"{net.id} head should be Q-values"


def test_dqn_head_width_matches_spec(models):
    q = models.MlpCritic(STATE, ACTION, NUM_TASKS)
    assert q.head.layers[-1].units == N_ACTIONS

    dqn = get_spec("dqn", N_ACTIONS)
    for net in dqn.networks:
        assert [h.units for h in net.heads] == [N_ACTIONS]
    # ddqn shares DQN's architecture.
    assert get_spec("ddqn", N_ACTIONS) == dqn


# --- Agent-level structure (which networks each agent builds) ---------------


def test_sac_networks_match_agent_attributes():
    """SAC's panel must list exactly the networks the SAC agent builds."""
    from mariha.rl.sac import SAC

    src = inspect.getsource(SAC.__init__)
    expected = {"actor", "critic1", "critic2", "target_critic1", "target_critic2"}
    for attr in expected:
        assert f"self.{attr}" in src, f"SAC no longer builds self.{attr}"
    assert {n.id for n in get_spec("sac", N_ACTIONS).networks} == expected


def test_dqn_networks_match_agent_attributes():
    """DQN's panel must list exactly the networks the DQN agent builds."""
    from mariha.rl.dqn import DQN

    src = inspect.getsource(DQN.__init__)
    for attr in ("q_network", "target_q_network"):
        assert f"self.{attr}" in src, f"DQN no longer builds self.{attr}"
    assert len(get_spec("dqn", N_ACTIONS).networks) == 2


def test_ppo_is_single_network():
    """PPO's panel must be the single shared actor-critic network."""
    from mariha.rl.ppo import PPO

    src = inspect.getsource(PPO.__init__)
    assert "self.model" in src, "PPO no longer builds self.model"
    assert len(get_spec("ppo", N_ACTIONS).networks) == 1


def test_link_endpoints_reference_known_networks():
    """Every Link must connect networks that actually exist in its panel."""
    for model in ("ppo", "sac", "dqn"):
        arch = get_spec(model, N_ACTIONS)
        ids = {n.id for n in arch.networks}
        for link in arch.links:
            assert link.src in ids, f"{model}: link src '{link.src}' unknown"
            assert link.dst in ids, f"{model}: link dst '{link.dst}' unknown"
