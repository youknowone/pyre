# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=go
# A BINARY_OP whose left operand is a builtin int and whose right operand is a
# user instance.  `int.__add__` answers NotImplemented, so the protocol runs
# `Acc.__radd__`.  That body mutates its receiver and returns a value that
# depends on the mutation; both the returned total and the mutation count have
# to survive compilation.
#
# Deterministic, terminating, prints PASS or a FAIL naming the site.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 60000


class Acc:
    def __init__(self):
        self.n = 0

    def __radd__(self, o):
        self.n += 1
        return self.n + o


def go(n, c):
    total = 0
    for i in range(n):
        total += i + c
    return total


def main():
    c = Acc()
    total = go(N, c)
    expected = N * N
    if total != expected:
        print(f"FAIL total {total}, expected {expected}")
        return 1
    if c.n != N:
        print(f"FAIL the reflected dunder ran {c.n} times, not {N}")
        return 1
    print("PASS builtin-left reflected dunder mutation")
    return 0


import sys

sys.exit(main())
