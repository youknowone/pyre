# CPython-suite gap: comprehension tests cannot exercise pyre's FOR_ITER JIT gate.
# parity-tests reason: this guards call-bearing comprehension trace admission.

import random

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

print("OK")
