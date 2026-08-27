# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=go
# A `BINARY_OP` dunder that COMMITS and then answers `NotImplemented`.
#
# The entry admits a body on the strength of its rewind, and the rewind is a
# record-time cut that is legal only while the region has applied nothing.  So
# the property that has to hold is "this body does not commit before the entry
# learns its result" -- and that is a property of the path the body walks, not
# of anything its code object spells.  Two static filters tried to stand in for
# it and this file is why neither can.
#
#   * A scan of `co_names` for `NotImplemented` cannot be sound.  The singleton
#     is an ordinary object: `L.__add__` below returns it through `self.ni`, so
#     the body's names are `('x', 'ni')` and hold no trace of it at all.  A
#     module-level alias, a freevar and a parameter default all do the same.
#   * A scan for a nested CALL cannot be sound either.  `self.x + o` enters a
#     nested callee's sub-walk -- `Inner.__add__` runs and commits -- with no
#     CALL opcode anywhere in `L.__add__`.  That is also, exactly, the shape of
#     the body this entry exists to admit (`return self.v + o.v` over ints),
#     so no static test separates them.
#
# What the abort costs is not a double-apply: `commits` below stays exactly N.
# It is the iteration's own contribution, dropped silently -- the answer came
# back 2099965 against 2100000, five aborts, thirty-five short.
#
# Deterministic, terminating, prints PASS or a FAIL naming the site.
N = 200000


class Inner:
    def __init__(self):
        self.n = 0

    def __add__(self, o):
        self.n += 1
        return 5


class L:
    def __init__(self):
        self.x = Inner()
        self.ni = NotImplemented

    def __add__(self, o):
        self.x + o
        return self.ni


class R:
    def __radd__(self, o):
        return 7


def go(n, l, r):
    total = 0
    for i in range(n):
        total += l + r
    return total


def main():
    l = L()
    r = R()
    total = go(N, l, r)
    if total != 7 * N:
        lost = (7 * N - total) // 7
        print(f"FAIL {lost} of {N} iterations lost their contribution ({total})")
        return 1
    if l.x.n != N:
        print(f"FAIL the committing dunder ran {l.x.n} times, not {N}")
        return 1
    print("PASS binop dunder commit-then-NotImplemented")
    return 0


import sys

sys.exit(main())
