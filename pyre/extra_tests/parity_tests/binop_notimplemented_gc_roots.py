# CPython-suite gap: no test returns NotImplemented from a dunder that collects.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""Operand dispatch must survive a dunder that returns ``NotImplemented``.

Both halves of the binary protocol run arbitrary Python before the operands
are used again: the forward call is tried first, and only when it answers
``NotImplemented`` does the reflected call — or the error path that names the
operand types — read them a second time.
"""

import gc


def churn():
    garbage = [[index, index + 1] for index in range(20000)]
    assert len(garbage) == 20000
    gc.collect()


class Reflected:
    """Answers NotImplemented so the reflected method is never the last word."""

    def __radd__(self, other):
        churn()
        return NotImplemented

    def __mul__(self, other):
        churn()
        return NotImplemented

    def __ror__(self, other):
        churn()
        return NotImplemented


for _ in range(10):
    # `list.__iadd__`: the bug-compat branch tries `Reflected.__radd__` first,
    # so the in-place method is entered with a receiver that has since moved.
    values = [1, 2, 3]
    try:
        values += Reflected()
    except TypeError as exc:
        assert "not iterable" in str(exc), exc
    else:
        raise AssertionError("list += Reflected() should not succeed")
    assert values == [1, 2, 3], values

    # `mul`: the forward `Reflected.__mul__` runs, declines, and the sequence
    # error path then has to name both operand types.
    values = [1, 2, 3]
    try:
        Reflected() * values
    except TypeError as exc:
        assert "can't multiply sequence" in str(exc), exc
    else:
        raise AssertionError("Reflected() * list should not succeed")

    # The same window with a dict operand, whose header also relocates.
    mapping = {"a": 1}
    try:
        mapping |= Reflected()
    except TypeError:
        pass
    assert mapping == {"a": 1}, mapping

print("OK")
