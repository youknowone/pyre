# A `__index__` that writes back to its own instance, consumed by `range()`.
#
# `functional.py:465` applies `space.index` once per bound and
# `descroperation.py:607 _index` calls the user method exactly once, so the
# counter must equal the iteration count. The specialization runs the callee
# during recording while keeping the caller pinned at the `range` CALL, so a
# guard failure resumes by re-entering that CALL and calls `__index__` again --
# the reason a body carrying a store is not admitted at all. The `& 1` makes
# the range alternate empty and non-empty, which is what makes the emptiness
# guard flip and the second call observable.
N = 20000


class Alt:
    def __init__(self):
        self.n = 0

    def __index__(self):
        self.n += 1
        return self.n & 1


def main():
    alt = Alt()
    total = 0
    for _ in range(N):
        total += len(range(alt))
    print(alt.n, total)


main()
# Expected: 20000 10000
