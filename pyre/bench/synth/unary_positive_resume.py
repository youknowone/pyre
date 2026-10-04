# pyre-check: max-pypy-ratio=4
# The measured reading stays under 4, so the ceiling is 4.
# The trip count puts pypy's execution above the startup-subtraction floor, so
# this ratio is a measurement. It collapses from the clamped reading rather
# than rising: at the old trip count the numerator was mostly pyre's fixed
# warmup, which the longer loop amortises, so the same code reads 2.1x where
# it read ~44x against the floor.
N = 200000000


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
