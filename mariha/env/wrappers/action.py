"""Reduced discrete action space wrapper for Super Mario Bros.

The NES controller exposes 9 buttons as a ``MultiBinary(9)`` space in
stable-retro.  Most button combinations are either redundant or meaningless
during Mario gameplay (e.g. SELECT, contradictory LEFT+RIGHT).

This wrapper maps a ``Discrete(9)`` action index to a specific NES button
combination, exposing only the subset of actions an agent actually needs.

NES button layout (stable-retro index order)::

    Index: 0  1     2       3      4   5     6     7      8
    Button: B  null  SELECT  START  UP  DOWN  LEFT  RIGHT  A

Action set (9 actions):

    0: NOOP
    1: RIGHT
    2: RIGHT + A   (jump right)
    3: RIGHT + B   (run right)
    4: RIGHT + A + B  (run-jump right)
    5: LEFT
    6: LEFT + A    (jump left)
    7: LEFT + B    (run left)
    8: A           (jump in place)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Button index constants (stable-retro NES layout)
# ---------------------------------------------------------------------------

_B = 0
_SELECT = 2
_START = 3
_UP = 4
_DOWN = 5
_LEFT = 6
_RIGHT = 7
_A = 8

_N_NES_BUTTONS = 9  # stable-retro MultiBinary size for NES

# ---------------------------------------------------------------------------
# Action map: discrete index → NES button array
# ---------------------------------------------------------------------------

# Each entry is a list of button indices that should be pressed for that action.
_ACTION_MAP: list[list[int]] = [
    [],                    # 0: NOOP
    [_RIGHT],              # 1: RIGHT
    [_RIGHT, _A],          # 2: RIGHT + A
    [_RIGHT, _B],          # 3: RIGHT + B
    [_RIGHT, _A, _B],      # 4: RIGHT + A + B
    [_LEFT],               # 5: LEFT
    [_LEFT, _A],           # 6: LEFT + A
    [_LEFT, _B],           # 7: LEFT + B
    [_A],                  # 8: A
]

N_ACTIONS = len(_ACTION_MAP)

ACTION_NAMES: list[str] = [
    "NOOP",
    "RIGHT",
    "RIGHT+A",
    "RIGHT+B",
    "RIGHT+A+B",
    "LEFT",
    "LEFT+A",
    "LEFT+B",
    "A",
]

def _buttons_to_array(buttons: list[int]) -> np.ndarray:
    """Convert a list of pressed button indices to a MultiBinary array.

    Args:
        buttons: Indices of buttons to press.

    Returns:
        ``np.ndarray`` of shape ``(9,)`` with dtype ``int8``.
    """
    arr = np.zeros(_N_NES_BUTTONS, dtype=np.int8)
    for idx in buttons:
        arr[idx] = 1
    return arr


# Pre-compute the button arrays once for efficiency.
_ACTION_ARRAYS: list[np.ndarray] = [_buttons_to_array(b) for b in _ACTION_MAP]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class ActionWrapper:
    """Replace the MultiBinary NES action space with a Discrete(9) space.

    Args:
        env: The environment to wrap (``SceneEnv`` or any compatible env).
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        return self._env.reset(**kwargs)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step with a discrete action index.

        Args:
            action: Integer action index in ``[0, N_ACTIONS)``.

        Returns:
            Standard gymnasium ``(obs, reward, terminated, truncated, info)`` tuple.
        """
        return self._env.step(_ACTION_ARRAYS[int(action)])

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
