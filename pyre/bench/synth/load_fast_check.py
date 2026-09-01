# pyre-check: max-pypy-ratio=4.4
# pyre-check: skip-cpython
# The loop body is two integer adds plus one read of a local the compiler
# cannot prove is bound, so LOAD_FAST_CHECK's per-iteration cost is most of
# what this fixture measures.  cpython is an order of magnitude slower on it
# and is not gated on; only pypy is.
# N is sized so pypy's own execution stays well clear of the ratio gate's
# minimum baseline: under that bar the gate divides by the measurement floor
# and the printed ratio reads process startup rather than this loop.
N = 88883772


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
