# pyre-check: max-pypy-ratio=3.3
# The ubuntu fleet reads this fixture in two modes: pypy executes the loop in
# 0.078s on one half of the runners and 0.027-0.05s on the other, for a pyre
# execution time that holds at 0.09-0.11s across both.  The ratio is therefore
# 1.3x or 2.1-2.8x depending on which runner draws the job, and the ceiling is
# the 2.8x maximum plus 15%, rounded up — set above the band rather than
# tightened onto the faster baseline.  Its 0.55x derived floor leaves the 1.3x
# reading well inside the band.  Windows' 6.0x display was ungated because that
# pypy execution baseline was clamped to the measurement floor.
# pyre-check: skip-cpython
# cpython 2.68s vs pyre 0.24s (11.2x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears the measurement floor on the slower
# half of the fleet; below the floor the ratio gate divides by the floor and
# reads startup rather than this loop.
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
