# pyre-check: gate=1
"""A builtin's other operand outlives the argument it coerces first.

`operator.length_hint`, `array.index`, `sorted` and `ContextVar.set` each run
Python for one argument — `__index__`, `__bool__`, or a mapping update — while
still holding another that the caller may have passed as a list or a dict.
"""

import array
import contextvars
import gc
import operator

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Idx:
    def __init__(self, n):
        self.n = n

    def __index__(self):
        gc.collect()
        churn()
        return self.n


class Truth:
    def __bool__(self):
        gc.collect()
        churn()
        return False


assert operator.length_hint([1, 2, 3], Idx(7)) == 3
assert operator.length_hint({"a": 1}, Idx(7)) == 1

numbers = array.array("i", [1, 2, 3, 4, 5])
assert numbers.index(4, Idx(0)) == 3
try:
    found = numbers.index([4], Idx(0))
except (ValueError, TypeError):
    found = "absent"
assert found == "absent", found

assert sorted([[3], [1], [2]], key=lambda x: x[0], reverse=Truth()) == [[1], [2], [3]]
assert sorted([[3], [1]], reverse=Truth()) == [[1], [3]]

variable = contextvars.ContextVar("gate")
variable.set([1, 2, 3])
for _ in range(20):
    churn()
    gc.collect()
token = variable.set([4, 5, 6])
gc.collect()
churn()
assert token.old_value == [1, 2, 3], token.old_value
variable.reset(token)
assert variable.get() == [1, 2, 3], variable.get()
