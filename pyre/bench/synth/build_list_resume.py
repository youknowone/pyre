N = 2000000


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
