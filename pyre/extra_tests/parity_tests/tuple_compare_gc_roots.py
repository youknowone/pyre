# CPython-suite gap: the vendored suite reaches this shape only by accident --
# `test.test_datetime` crashed on Linux and passed on macOS at the same commit,
# so whether it covers the walk depends on where a collection happens to land.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""Tuple comparison must survive an element ``__eq__`` that collects.

``_compare_tuples`` walks both operands element by element, and each step
runs ``space.eq_w`` on one pair of items -- arbitrary Python, so a collection
point.  The next step reads both tuples again, and the pair the walk stops on
is read once more to produce the ordering answer.
"""

import gc


def churn():
    garbage = [[index, index + 1] for index in range(8000)]
    assert len(garbage) == 8000
    gc.collect()


class Cell:
    """Comparisons that collect, so every element boundary moves both operands."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        churn()
        return self.value == other.value

    def __lt__(self, other):
        churn()
        return self.value < other.value

    def __gt__(self, other):
        churn()
        return self.value > other.value


for _ in range(5):
    # Array-backed tuples: the walk crosses a collection between every pair, so
    # items 2 and 3 are fetched out of tuples that have since moved.
    left = (Cell(0), Cell(1), Cell(2), Cell(3))
    right = (Cell(0), Cell(1), Cell(2), Cell(3))
    assert left == right
    assert not (left != right)

    # Arity two with object slots is the specialised representation, which
    # compares its two raw slots in a loop of its own.
    assert (Cell(4), Cell(5)) == (Cell(4), Cell(5))
    assert (Cell(4), Cell(5)) != (Cell(4), Cell(6))

    # The walk stops on an unequal pair, and the ordering answer comes from
    # reading that pair back out of the operands.
    assert (Cell(0), Cell(1), Cell(2)) < (Cell(0), Cell(1), Cell(3))
    assert (Cell(0), Cell(9), Cell(2)) > (Cell(0), Cell(1), Cell(3))

    # Every shared item agrees, so the answer is the size comparison, taken
    # from the lengths read before the walk.
    assert (Cell(0), Cell(1)) < (Cell(0), Cell(1), Cell(2))

print("OK")
