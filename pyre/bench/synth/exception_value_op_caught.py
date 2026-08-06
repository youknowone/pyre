# pyre-check: max-pypy-ratio=52
# pyre-check: min-pypy-ratio=3
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (25.1x), rounded up, and the floor is half the fastest
# (6.0x) — a derived floor of ceiling/5 would sit above it.
N = 100000


# Custom operands route the binary `+` and rich-compare `<` through the
# residual value helpers (jit_binary_value_from_tag /
# jit_compare_value_from_tag), each followed by a GuardNoException.  Both
# operators raise every iteration, so the JIT deopts into the blackhole on
# the top frame after the residual call.  The raising op sits in a
# try-block, so the snapshot resumes at the call's own catch_exception; the
# liveness read for the active boxes must use that SAME post-call `-live-`
# as the snapshot pc, or the blackhole decoder consumes a different box
# count than the encoder wrote.
class Boom:
    def __add__(self, other):
        raise ValueError("add")

    def __lt__(self, other):
        raise ValueError("lt")


def main():
    b = Boom()
    acc = 0
    i = 0
    while i < N:
        try:
            b + 1
        except ValueError:
            acc = acc + 1
        try:
            b < 1
        except ValueError:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
