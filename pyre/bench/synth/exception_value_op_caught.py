# No `max-pypy-ratio`: pypy's exec time here sits at the startup-subtraction
# floor, so the printed ratio is a quotient by a clamped denominator rather
# than a measurement, and which side of the floor it lands on flips run to run
# -- which decides whether the ceiling is applied at all.  A ceiling read off
# such a number gates noise; 175 of the 463 fixtures state none for the same
# reason, and the ratio is still printed on every run.  This fixture's gates
# are its three-backend jit-stats baselines.
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
