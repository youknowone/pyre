# pyre-check: gate=1
# `W_Deque.mul` builds its answer by extending a `maxlen`-bounded copy, and
# every `append` behind that `extend` runs `trimleft`.  So repeating a bounded
# deque holds `maxlen` items however large the count is, and the only bound on
# the count itself is `ovfcheck(self.len * num)` -- a machine-signed product,
# which is why the two-item case below raises where the one-item case does not
# even look.  A version that materialises the whole product before trimming
# runs out of memory on the large counts here instead of answering.
from collections import deque

from testutils import assert_raises

assert deque([1], maxlen=1) * 20_000_000 == deque([1], maxlen=1)
assert deque([1, 2, 3], maxlen=5) * 1_000_000 == deque([2, 3, 1, 2, 3], maxlen=5)
assert 1_000_000 * deque([1, 2, 3], maxlen=5) == deque([2, 3, 1, 2, 3], maxlen=5)
assert deque([1, 2, 3], maxlen=4) * 1_000_000 == deque([3, 1, 2, 3], maxlen=4)
assert_raises(MemoryError, lambda: deque([1, 2], maxlen=2) * (2**62))

d = deque([1], maxlen=1)
d *= 20_000_000
assert d == deque([1], maxlen=1)

d = deque([1, 2, 3], maxlen=5)
d *= 1_000_000
assert d == deque([2, 3, 1, 2, 3], maxlen=5)


def imul_overflow():
    d = deque([1, 2], maxlen=2)
    d *= 2**62


assert_raises(MemoryError, imul_overflow)

# An unbounded deque keeps the ordinary product, and the counts that short
# circuit are unaffected.
assert deque([1, 2]) * 3 == deque([1, 2, 1, 2, 1, 2])
assert deque([1, 2, 3]) * 0 == deque([])
assert deque([1, 2, 3], maxlen=5) * 1 == deque([1, 2, 3], maxlen=5)
assert deque([1, 2, 3], maxlen=5) * -5 == deque([])
