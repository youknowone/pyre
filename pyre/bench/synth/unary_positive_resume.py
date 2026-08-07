# pyre-check: max-pypy-ratio=5
# The trip count puts pypy's execution above the startup-subtraction floor, so
# this ratio is a measurement. It collapses from the clamped reading rather
# than rising: at the old trip count the numerator was mostly pyre's fixed
# warmup, which the longer loop amortises, so the same code reads 2.1x where
# it read ~44x against the floor. The ceiling is twice the slowest of the
# three backends observed (2.1x on wasm); the previous 90 gated nothing.
N = 110000000


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
