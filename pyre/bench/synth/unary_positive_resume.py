# pyre-check: max-pypy-ratio=90
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (44.3x), rounded up.
N = 2000000


def main():
    total = 0
    i = 0
    while i < N:
        # `+x` compiles to CALL_INTRINSIC_1 INTRINSIC_UNARY_POSITIVE.
        # The varying operands make the loop's guards deopt, so the
        # blackhole walks the portal jitcode through CALL_INTRINSIC_1 and
        # computes `+value` directly on resume instead of aborting the
        # trace.
        total += (+i + +(i + 1)) & 7
        i += 1
    print(total)


main()
