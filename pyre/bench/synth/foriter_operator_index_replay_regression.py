# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=<module>
# `operator.index(x)` reaches `space.index`, whose first test is
# `is_int_or_long`: an int is returned as-is, before any `__index__` lookup, so
# that call runs no user code and is replay-safe.  Every other argument
# dispatches through `__index__`, which IS user code — so the replay-safe class
# must observe the argument, not merely pin the callable.
#
# `helper` is admitted into the surrounding FOR_ITER body.  The trailing opaque
# `id` call makes the first inline sub-walk abort and replay `helper`.  The int
# call is the arm being admitted; the object call must stay opaque.  Admitting
# the object call too would let the replay run `__index__` a second time, and
# `hits` would read N + 1.

from operator import index as _index

N = 5000
hits = [0]


class C:
    def __index__(self) -> int:
        hits[0] += 1
        return 3


def helper(obj, n):
    a = _index(n)
    b = _index(obj)
    id(obj)
    return a + b


obj = C()
total = 0
for _ in range(N):
    total += helper(obj, 1)

assert hits[0] == N, f"__index__ ran {hits[0]} times, expected {N}"
assert total == 4 * N, f"total {total}, expected {4 * N}"
print("PASS")
