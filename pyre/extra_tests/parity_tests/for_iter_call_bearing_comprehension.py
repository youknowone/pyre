# CPython-suite gap: comprehension tests cannot exercise pyre's FOR_ITER JIT gate.
# parity-tests reason: this guards call-bearing comprehension trace admission.

import random
from operator import itemgetter

# A comprehension whose body calls a user Python function accumulates through
# LIST_APPEND. `randrange` executes `_operator.index` on its bound, and while
# that was booked as a body effect the mid-body walk abort refused the consumed
# item's delivery, the legacy replay resumed at the next iteration, and the
# element was lost. Only lengths are asserted, so the check holds whatever
# values the generator produces.
random.seed(1234)
for trial in range(400):
    size = random.randrange(50)
    data = [random.randrange(25) for _ in range(size)]
    assert len(data) == size, (trial, size, len(data))

# The same shape spelled as a statement loop, and one where the accumulator is
# a local rather than the comprehension's own temporary.
for trial in range(200):
    size = trial % 50
    collected = []
    for _ in range(size):
        collected.append(random.randrange(25))
    assert len(collected) == size, (trial, size, len(collected))


class Payload:
    def __init__(self, value):
        self.value = value


def handled(items):
    out = []
    seen = []
    for index in items:
        try:
            items[index + 100]
        except IndexError:
            out.extend([str(Payload(value * 3 + 1)) for value in range(1)])
        seen.append(len(range(index)))
        out.append(index)
    return len(out), len(seen)


items = [index % 5 for index in range(60)]
for _ in range(400):
    assert handled(items) == (120, 60)

# A guard switching itemgetter from its scalar arm to its internal
# comprehension must retain the item already consumed by FOR_ITER.
single = itemgetter(0)
value = ("B", -260)
for _ in range(20_000):
    single(value)
multiple = itemgetter(1, 0)
for _ in range(20_000):
    assert multiple(value) == (-260, "B")

print("OK")
