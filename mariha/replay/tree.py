"""Priority trees for prioritized experience replay.

``SumTree`` and ``SegmentTree`` are data structures that support O(log n)
priority-based sampling used by ``PrioritizedReplayBuffer`` and
``PrioritizedExperienceReplay``.

Ported verbatim from COOM (no external dependencies beyond NumPy).
"""

import numpy as np
from typing import Optional, Union


class SumTree:
    """Binary sum tree for prioritized replay sampling.

    Based on the implementation by Morvan Zhou.
    Each leaf stores a priority score; internal nodes store subtree sums.

    Args:
        capacity: Maximum number of experiences (leaf nodes).
    """

    data_pointer = 0

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority: float, data) -> None:
        """Add an experience with the given priority."""
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index: int, priority: float) -> None:
        """Update a leaf's priority and propagate the change upward."""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v: float):
        """Return (leaf_index, priority, data) for a given cumulative value."""
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self) -> float:
        """Sum of all priorities (root node value)."""
        return self.tree[0]


class SegmentTree:
    """Segment tree for O(log n) prefix-sum queries.

    Used by ``PrioritizedExperienceReplay`` for efficient importance-sampling
    weight calculation.

    Args:
        size: Number of elements to store.
    """

    def __init__(self, size: int) -> None:
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._value = np.zeros([bound * 2])
        self._compile()

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self._value[index + self._bound]

    def __setitem__(
        self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]
    ) -> None:
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(0 <= index) and np.all(index < self._size)
        _setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return the sum of elements in ``[start, end)``."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return _reduce(self._value, start + self._bound - 1, end + self._bound)

    def get_prefix_sum_idx(
        self, value: Union[float, np.ndarray]
    ) -> Union[int, np.ndarray]:
        """Return the minimum index i such that sum(arr[0..i]) >= value."""
        assert np.all(value >= 0.0) and np.all(value < self._value[1])
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        _setitem(f64, i64, f64)
        _setitem(f64, i64, f32)
        _reduce(f64, 0, 1)
        _get_prefix_sum_idx(f64, 1, f64)
        _get_prefix_sum_idx(f32, 1, f64)


# ---------------------------------------------------------------------------
# Tree helpers (vectorised NumPy — avoids Python loops in hot path)
# ---------------------------------------------------------------------------


def _get_prefix_sum_idx(
    value: np.ndarray, bound: int, sums: np.ndarray
) -> np.ndarray:
    index = np.ones(value.shape, dtype=np.int64)
    while index[0] < bound:
        index *= 2
        lsons = sums[index]
        direct = lsons < value
        value -= lsons * direct
        index += direct
    index -= bound
    return index


def _reduce(tree: np.ndarray, start: int, end: int) -> float:
    result = 0.0
    while end - start > 1:
        if start % 2 == 0:
            result += tree[start + 1]
        start //= 2
        if end % 2 == 1:
            result += tree[end - 1]
        end //= 2
    return result


def _setitem(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = tree[index * 2] + tree[index * 2 + 1]
