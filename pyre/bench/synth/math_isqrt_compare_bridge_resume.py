# pyre-check: max-pypy-ratio=19
# pyre-check: min-pypy-ratio=1.72
# A comparison helper (like test_math.testIsqrt's assertLessEqual/assertLess)
# runs first on exact ints and then on bignums. The `<=`/`<` inside the helper
# callee has CompareOp class/value guards that must attach a bridge at the
# comparison's opcode-start resume point instead of deopting every later call.
# unittest is unavailable on the wasm backend (no os module), so the helper is
# a plain class rather than unittest.TestCase.

import math


class Cmp:
    def le(self, a, b):
        assert a <= b

    def lt(self, a, b):
        assert a < b


cmp = Cmp()
BIG = 10**100
checksum = 0

for value in list(range(5000)) + [BIG] * 80000:
    root = math.isqrt(value)
    assert type(root) is int
    cmp.le(root * root, value)
    cmp.lt(value, (root + 1) * (root + 1))
    checksum = (checksum + (root & 65535)) % 1000000007

print(checksum)
