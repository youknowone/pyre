# pyre-check: max-pypy-ratio=14
# pyre-check: skip-cpython
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
# 153183341 puts pypy at 258ms, and the two requirements do not both fit: a
# count small enough for cpython to finish inside its reference timeout leaves
# pypy an order of magnitude under the floor. cpython is dropped deliberately
# rather than by spending the whole timeout to discover the same drop.
N = 153183341


def main():
    total = 0
    i = 0
    while i < N:
        # A 5-element list display compiles to BUILD_LIST 5 (argc > 3),
        # the arbitrary-arity form that the fixed three-slot build_list_fn
        # cannot cover.  The varying elements make the loop's guards deopt,
        # so the blackhole walks the portal jitcode through BUILD_LIST and
        # builds the list directly on resume instead of aborting the trace.
        lst = [i, i + 1, i + 2, i + 3, i + 4]
        total += (lst[0] + lst[4]) & 7
        i += 1
    print(total)


main()
