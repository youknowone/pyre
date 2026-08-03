"""`memoryview.count` / `.index` survive a user `__eq__` that allocates.

Element comparison runs arbitrary Python between iterations, so the view, the
sought value and the iterator must outlive a collection triggered from inside
`__eq__`.  `StopIteration.__init__` has the same shape: it stores `value`
before delegating, whose `args` allocation is a collection point.
"""

import gc


class Churn:
    """Compares equal to `target`, allocating heavily on every comparison."""

    def __init__(self, target):
        self.target = target
        self.junk = []

    def __eq__(self, other):
        self.junk = [[object() for _ in range(64)] for _ in range(40)]
        gc.collect()
        return other == self.target


view = memoryview(bytearray(range(64)) * 8)

assert view.count(Churn(7)) == 8
assert view.count(Churn(200)) == 0
assert view.index(Churn(7)) == 7
assert view.index(Churn(7), 8) == 71

try:
    view.index(Churn(200))
except ValueError as error:
    assert str(error) == "memoryview.index(x): x not found", error
else:
    raise AssertionError("index() must report a missing value")

for step in range(20000):
    stop = StopIteration()
    stop.__init__([step, step + 1])
    assert stop.value == [step, step + 1], (step, stop.value)
    assert stop.args == ([step, step + 1],)

reused = StopIteration("first")
assert reused.value == "first"
reused.__init__()
assert reused.value is None
assert reused.args == ()

print("OK")
