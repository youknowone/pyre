# pyre-check: spec-folds=make_function,set_function_attribute
# pyre-check: max-pypy-ratio=4
# pyre-check: skip-cpython
# An annotated `def` in a hot FOR_ITER body: the SET_FUNCTION_ATTRIBUTE arm
# 3.14 reaches for far more often than the defaults one, and the one that
# decides whether anything else in the definition sequence folds.
#
# A return annotation alone compiles to two MAKE_FUNCTIONs -- one for the
# `__annotate__` closure PEP 649 defers the annotation to -- and a single
# `SET_FUNCTION_ATTRIBUTE annotate`, emitted BEFORE any defaults stamp. So this
# fixture's one attribute stamp IS the annotate arm: `spec-folds` firing here
# cannot be satisfied by the defaults arm the way it can in
# `foriter_make_function_body`, which is why the shape is worth its own
# fixture rather than an annotation added to that one.
#
# N is set by `FLOOR_GATE_MIN_BASELINE_S` rather than by the shape: folded, the
# iteration is a couple of instructions, and a ratio whose baseline sits under
# the floor -- 0.05s on darwin/linux, 0.15625s on windows, whose CPU accounting
# advances in 1/64s ticks -- is declined rather than judged. At 500000000
# dynasm's execution-only time reads 0.49-0.53s and pypy's 0.25-0.27s over five
# interleaved reps, so both clear the windows line by better than 1.5x. The
# ratio reads 1.79-2.08x against a ceiling of 4.
#
# `PYRE_FBW_NO_SPECIALIZE=set_function_attribute` on the same binary reads
# 30.5s against 0.52s -- 58x, and nowhere near the ceiling.
#
# `skip-cpython`: cpython needs tens of seconds at this size, well past
# `SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S`, so it would be dropped anyway --
# after spending that timeout on every run. pypy is the oracle the output is
# checked against.
N = 500000000


def main():
    total = 0
    for i in range(N):

        def add(value) -> int:
            return value + 1

        total += add(i)
    print(total)


main()
# Expected: 125000000250000000
