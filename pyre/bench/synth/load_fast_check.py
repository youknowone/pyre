# pyre-check: max-pypy-ratio=9.5
# pyre-check: skip-cpython
# cpython 2.68s vs pyre 0.24s (11.2x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 35553509


def main():
    acc = 0
    i = 0
    while i < N:
        # `i >= 0` is always true, so `x` is bound on every iteration, but
        # the compiler's definite-assignment analysis cannot prove the branch
        # is always taken — it emits LOAD_FAST_CHECK for the `acc + x` read.
        if i >= 0:
            x = i
        acc = acc + x
        i = i + 1
    print(acc)


main()
