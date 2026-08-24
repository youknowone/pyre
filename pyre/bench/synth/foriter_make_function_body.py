# pyre-check: spec-folds=make_function,set_function_attribute
# pyre-check: max-pypy-ratio=4
# pyre-check: skip-cpython
# MAKE_FUNCTION plus SET_FUNCTION_ATTRIBUTE in a hot FOR_ITER body. The default
# value forces the companion attribute initializer onto the definition path.
#
# `spec-folds` is what gates the subject. Both opcodes emit their effect inline
# -- the allocation and its `Function.__init__` stores, and then the single
# `SetfieldGc` the attribute flag names -- so the definition sequence
# virtualizes away entirely when the function does not escape, which is what
# the loop here does. What is left to time is the arithmetic, and no throughput
# number can distinguish "the fold fired" from "the loop was cheap anyway"; a
# census that says each fold fired at least once can.
#
# The ceiling still gates the sum, and N is what makes it judgeable.
# `FLOOR_GATE_MIN_BASELINE_S` declines a ratio whose baseline is under the
# measurement floor -- 0.05s on darwin/linux, 0.15625s on windows, whose CPU
# accounting advances in 1/64s ticks -- and with the definition folded away
# each iteration is a couple of instructions, so the floor is what sets the
# size rather than any property of the shape. At 500000000 dynasm's
# execution-only time reads 0.43-0.69s and pypy's 0.28-0.35s over eight
# interleaved reps, so both clear the windows line by better than 2x and the
# darwin/linux one by better than 5x. The ratio between them reads 1.46-2.04x
# across those reps; the ceiling of 4 clears the highest of them by 2x. Putting
# either fold back on the residual is nowhere near that bar:
# PYRE_FBW_NO_SPECIALIZE=set_function_attribute reads 28.4ns per iteration
# against 0.8ns folded, and naming both reads 225ns -- 35x and 276x, measured
# interleaved at a fifth of the size below.
#
# `skip-cpython`: cpython needs ~40s at this size, eight times
# `SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S`, so it would be dropped anyway --
# after spending that timeout on every run. pypy is the oracle the output is
# checked against.
N = 500000000


def main():
    total = 0
    for i in range(N):

        def add(value=i):
            return value + 1

        total += add()
    print(total)


main()
# Expected: 125000000250000000
