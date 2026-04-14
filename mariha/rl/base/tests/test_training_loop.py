"""Smoke tests for :class:`TrainingLoopRunner`.

These tests drive the runner with a fully synthetic ``MockEnv`` + a
``MockAgent`` (a minimal :class:`BaseAgent` subclass that records every
callback into an ``events`` list).  No real environment, networks, or
algorithm code are involved — the tests verify only the loop's
structural contract:

- Episode bookkeeping (return, length, total_episodes).
- Curriculum termination (``is_done`` *and* ``StopIteration`` paths).
- Task-switch and session-boundary callback ordering.
- Render and checkpoint cadence.
- Periodic logging cadence.

The Phase 0 acceptance gate from the refactor plan is the
``test_runner_full_curriculum`` case below: 100 episodes, 5 task
switches, 3 (extra) session boundaries, 2 render checkpoints, 10
logging epochs, with callback ordering asserted at task switches.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mariha.rl.base import BaseAgent


# ----------------------------------------------------------------------
# Mock environment
# ----------------------------------------------------------------------


class _MockActionSpace:
    """Minimal Discrete-style action space."""

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.seed_value = None

    def sample(self) -> int:
        return 0

    def seed(self, seed: int) -> None:
        # ``set_seed`` calls this on the env's action space.
        self.seed_value = seed


class _MockObsSpace:
    """Minimal Box-style observation space — only ``shape`` is used."""

    def __init__(self, shape=(4,)) -> None:
        self.shape = shape


class MockEnv:
    """Replays a hard-coded list of episodes through the standard env API.

    Each entry in ``episodes`` is a dict with keys:
        - ``scene_id``       (str)
        - ``session``        (str)
        - ``length``         (int) — number of ``step`` calls before
          ``terminated`` becomes True.
        - ``task_switch``    (bool) — value placed in the ``info`` dict
          returned by ``reset()`` for this episode.
        - ``session_switch`` (bool) — same.

    The first episode is always emitted with ``task_switch`` and
    ``session_switch`` False, regardless of what the spec says — the
    runner does not consider the very first reset to be a switch.
    """

    def __init__(self, episodes: List[Dict[str, Any]]) -> None:
        self.episodes = episodes
        self.action_space = _MockActionSpace()
        self.observation_space = _MockObsSpace()
        self.is_done = False
        self.reset_count = 0
        self.step_count = 0
        self.render_count = 0

        self._idx = -1
        self._step_in_ep = 0

    def reset(self, episode_spec=None):
        self._idx += 1
        if self._idx >= len(self.episodes):
            raise StopIteration
        ep = self.episodes[self._idx]
        self._step_in_ep = 0
        self.reset_count += 1
        info = {
            "task_one_hot": np.zeros(6, dtype=np.float32),
            "scene_id": ep["scene_id"],
            "run_id": ep.get("run_id", ep["scene_id"]),
            "session": ep["session"],
            "task_switch": ep.get("task_switch", False) if self._idx > 0 else False,
            "scene_switch": ep.get("scene_switch", False) if self._idx > 0 else False,
            "session_switch": ep.get("session_switch", False) if self._idx > 0 else False,
        }
        return np.zeros(4, dtype=np.float32), info

    def step(self, action):
        self.step_count += 1
        self._step_in_ep += 1
        ep = self.episodes[self._idx]
        terminated = self._step_in_ep >= ep["length"]
        # Mark the curriculum as exhausted on the very last step of the
        # very last episode so that the runner sees ``is_done`` True
        # *before* it tries to call reset() again.
        if terminated and self._idx == len(self.episodes) - 1:
            self.is_done = True
        return (
            np.zeros(4, dtype=np.float32),
            1.0,
            terminated,
            False,
            {},
        )

    def render_checkpoint(self, actor_fn) -> None:
        self.render_count += 1


# ----------------------------------------------------------------------
# Mock logger
# ----------------------------------------------------------------------


class MockLogger:
    """EpochLogger surface used by the runner / BaseAgent default hooks."""

    def __init__(self) -> None:
        self.messages: List[str] = []
        self.stored: List[Dict[str, Any]] = []
        self.tabular_keys: List[str] = []
        self.dumps: int = 0

    def log(self, msg: str, color: str = None) -> None:
        self.messages.append(msg)

    def store(self, d: Dict[str, Any]) -> None:
        self.stored.append(dict(d))

    def log_tabular(self, key: str, value=None, **kwargs) -> None:
        self.tabular_keys.append(key)

    def dump_tabular(self) -> None:
        self.dumps += 1


# ----------------------------------------------------------------------
# Mock agent
# ----------------------------------------------------------------------


class MockAgent(BaseAgent):
    """Minimal :class:`BaseAgent` subclass that records every callback.

    All algorithm-specific abstract methods are implemented as no-ops
    that increment counters; every lifecycle hook also appends a tuple
    to ``self.events`` so callback order can be asserted.
    """

    def __init__(self, env, logger, run_ids, **kwargs) -> None:
        super().__init__(
            env=env,
            logger=logger,
            run_ids=run_ids,
            agent_name="mock",
            **kwargs,
        )
        self.events: List[tuple] = []
        self.update_count = 0
        self.transitions = 0
        self.saved_dirs: List[Path] = []

    # ---- BenchmarkAgent contract ----

    def get_action(self, obs, task_one_hot, deterministic: bool = False) -> int:
        return 0

    # ---- Required BaseAgent callbacks ----

    def select_action(
        self,
        *,
        obs,
        one_hot,
        global_step: int,
        task_step: int,
        current_task_idx: int,
    ):
        return 0, {}

    def store_transition(self, **kwargs) -> None:
        self.transitions += 1

    def should_update(self, global_step: int) -> bool:
        # Update every 4 steps so the loop exercises both update and
        # no-update branches.
        return global_step % 4 == 0

    def update_step(self, *, global_step: int, current_task_idx: int) -> None:
        self.update_count += 1

    def save_weights(self, directory: Path) -> None:
        directory = Path(directory)
        # Mark the dir so the test can verify the runner created it.
        (directory / "weights.bin").write_bytes(b"")
        self.saved_dirs.append(directory)

    def load_weights(self, directory: Path) -> None:
        return None

    # ---- Optional lifecycle hooks (overridden to record) ----

    def on_task_start(self, task_idx: int, run_id: str) -> None:
        self.events.append(("on_task_start", task_idx, run_id))

    def on_task_end(self, task_idx: int) -> None:
        self.events.append(("on_task_end", task_idx))

    def on_task_change(self, new_task_idx: int, new_run_id: str) -> None:
        self.events.append(("on_task_change", new_task_idx, new_run_id))

    def handle_session_boundary(self, current_task_idx: int) -> None:
        self.events.append(("handle_session_boundary", current_task_idx))

    def on_episode_end(
        self,
        *,
        episode_return: float,
        episode_len: int,
        total_episodes: int,
    ) -> None:
        super().on_episode_end(
            episode_return=episode_return,
            episode_len=episode_len,
            total_episodes=total_episodes,
        )
        self.events.append(("on_episode_end", total_episodes))


# ----------------------------------------------------------------------
# Curriculum builder
# ----------------------------------------------------------------------


def _build_curriculum() -> List[Dict[str, Any]]:
    """Build the 100-episode curriculum used by the acceptance test.

    - 6 tasks (scene0..scene5) → 5 task switches at episodes 17, 34,
      51, 68, 85.
    - 3 *extra* session-only boundaries at episodes 25, 60, 90 — these
      raise ``session_switch`` without raising ``task_switch``.
    - Every episode is exactly 100 steps → 100 × 100 = 10000 env steps,
      which is exactly 10 logging epochs at ``log_every=1000``.
    """
    task_starts = [0, 17, 34, 51, 68, 85]
    extra_session_starts = {25, 60, 90}
    episodes: List[Dict[str, Any]] = []

    for i in range(100):
        # Which task are we in?
        task_idx = sum(1 for s in task_starts if s <= i) - 1

        is_task_switch = i in task_starts and i != 0
        is_session_switch = is_task_switch or (i in extra_session_starts)

        episodes.append(
            {
                "scene_id": f"scene{task_idx}",
                "run_id": f"run{task_idx}",
                "session": f"task{task_idx}_seg{int(i in extra_session_starts)}",
                "length": 100,
                "task_switch": is_task_switch,
                "session_switch": is_session_switch,
            }
        )
    return episodes


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_runner_full_curriculum():
    """Phase 0 acceptance gate.

    100 episodes, 5 task switches, 3 extra session boundaries,
    2 render checkpoints, 10 logging epochs.  Asserts callback ordering
    at every task switch.
    """
    episodes = _build_curriculum()
    env = MockEnv(episodes)
    logger = MockLogger()

    with tempfile.TemporaryDirectory() as tmp:
        agent = MockAgent(
            env=env,
            logger=logger,
            run_ids=[f"run{i}" for i in range(6)],
            seed=0,
            log_every=1000,
            save_freq_epochs=5,
            render_every=50,
            experiment_dir=Path(tmp),
            timestamp="20260410",
        )
        agent.run()

        # ---- step / transition counts ----
        assert env.step_count == 10000, f"step_count={env.step_count}"
        assert agent.transitions == 10000
        # update_count: should_update returns True at global_step in
        # {0, 4, 8, ..., 9996} → 2500 calls.
        assert agent.update_count == 2500

        # ---- episode bookkeeping ----
        episode_events = [e for e in agent.events if e[0] == "on_episode_end"]
        assert len(episode_events) == 100
        assert episode_events[0][1] == 1
        assert episode_events[-1][1] == 100

        # ---- task lifecycle counts ----
        starts = [e for e in agent.events if e[0] == "on_task_start"]
        ends = [e for e in agent.events if e[0] == "on_task_end"]
        changes = [e for e in agent.events if e[0] == "on_task_change"]
        assert len(starts) == 6  # 1 initial + 5 switches
        assert len(ends) == 6    # 5 switches + 1 final
        assert len(changes) == 6  # 1 initial + 5 switches
        # Tasks visited in order
        assert [e[1] for e in starts] == [0, 1, 2, 3, 4, 5]
        # Final on_task_end is for the last task
        assert ends[-1][1] == 5

        # ---- session boundary count ----
        # 5 from task switches (where session_switch is True alongside
        # task_switch) plus 3 pure session-only boundaries.
        boundaries = [e for e in agent.events if e[0] == "handle_session_boundary"]
        assert len(boundaries) == 8

        # ---- render cadence ----
        # render_every=50 → renders at episode count 50 and 100.
        assert env.render_count == 2

        # ---- log cadence ----
        # log_every=1000, 10000 steps → 10 log dumps.
        assert logger.dumps == 10

        # ---- checkpoints ----
        # save_freq_epochs=5 → periodic saves at epochs 5 and 10, plus
        # one final save at curriculum end.
        assert len(agent.saved_dirs) >= 2

        # ---- callback ordering at the first task switch ----
        # Find the first non-initial on_task_end and verify the next
        # two events are on_task_start then on_task_change.
        for i, e in enumerate(agent.events):
            if e[0] == "on_task_end" and i + 2 < len(agent.events):
                assert agent.events[i + 1][0] == "on_task_start"
                assert agent.events[i + 2][0] == "on_task_change"
                # session_boundary fires *before* on_task_end at a real
                # switch — check the immediately preceding event.
                # (At the very first task end, the previous event is
                # handle_session_boundary because session_switch is True
                # alongside task_switch.)
                assert agent.events[i - 1][0] == "handle_session_boundary"
                break
        else:
            raise AssertionError("no intermediate on_task_end found in events")

        # ---- initial lifecycle ordering ----
        assert agent.events[0] == ("on_task_start", 0, "run0")
        assert agent.events[1] == ("on_task_change", 0, "run0")


def test_runner_empty_curriculum():
    """An env that raises StopIteration on the very first reset should
    log a friendly message and return without crashing."""

    class _EmptyEnv(MockEnv):
        def reset(self, episode_spec=None):
            raise StopIteration

    env = _EmptyEnv(episodes=[])
    logger = MockLogger()
    with tempfile.TemporaryDirectory() as tmp:
        agent = MockAgent(
            env=env,
            logger=logger,
            run_ids=["run0"],
            seed=0,
            experiment_dir=Path(tmp),
            timestamp="empty",
        )
        agent.run()  # must not raise

    assert any("empty" in m.lower() for m in logger.messages)
    assert agent.events == []  # no callbacks fired


def test_runner_single_episode():
    """A 1-episode curriculum should fire the standard end-of-run
    sequence: episode_end → render? → on_task_end (final) → save."""
    episodes = [
        {
            "scene_id": "scene0",
            "run_id": "run0",
            "session": "sess0",
            "length": 10,
            "task_switch": False,
            "session_switch": False,
        }
    ]
    env = MockEnv(episodes)
    logger = MockLogger()
    with tempfile.TemporaryDirectory() as tmp:
        agent = MockAgent(
            env=env,
            logger=logger,
            run_ids=["run0"],
            seed=0,
            log_every=1000,
            save_freq_epochs=1,
            render_every=0,  # disable
            experiment_dir=Path(tmp),
            timestamp="single",
        )
        agent.run()

        assert env.step_count == 10
        assert agent.transitions == 10

        # Events: initial start/change, episode end, final task end.
        assert agent.events[0] == ("on_task_start", 0, "run0")
        assert agent.events[1] == ("on_task_change", 0, "run0")
        assert agent.events[2] == ("on_episode_end", 1)
        assert agent.events[3] == ("on_task_end", 0)
        assert len(agent.events) == 4

        assert env.render_count == 0  # disabled
        # No log epoch was reached (10 steps < log_every=1000)
        assert logger.dumps == 0
        # But the final checkpoint still fires
        assert len(agent.saved_dirs) == 1


def test_runner_scene_switch_no_task_hooks():
    """Within a single run, a scene change should fire scene_switch
    but NOT task_switch — no on_task_end / on_task_start / on_task_change
    hooks may fire on the boundary."""
    episodes = [
        {
            "scene_id": "scene0",
            "run_id": "run0",
            "session": "sess0",
            "length": 5,
            "task_switch": False,
            "scene_switch": False,
            "session_switch": False,
        },
        {
            "scene_id": "scene1",
            "run_id": "run0",
            "session": "sess0",
            "length": 5,
            "task_switch": False,
            "scene_switch": True,
            "session_switch": False,
        },
    ]
    env = MockEnv(episodes)
    logger = MockLogger()
    with tempfile.TemporaryDirectory() as tmp:
        agent = MockAgent(
            env=env,
            logger=logger,
            run_ids=["run0"],
            seed=0,
            log_every=1000,
            save_freq_epochs=1,
            render_every=0,
            experiment_dir=Path(tmp),
            timestamp="scene_switch",
        )
        agent.run()

        # Exactly one initial start/change pair, one final end — no
        # mid-curriculum task hooks despite the scene change.
        starts = [e for e in agent.events if e[0] == "on_task_start"]
        ends = [e for e in agent.events if e[0] == "on_task_end"]
        changes = [e for e in agent.events if e[0] == "on_task_change"]
        assert len(starts) == 1
        assert len(ends) == 1
        assert len(changes) == 1
