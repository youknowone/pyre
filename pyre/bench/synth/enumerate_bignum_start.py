# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=1` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# enumerate(iterable, start) accepts an arbitrary-precision start past i64,
# activating the bigint index slot instead of raising OverflowError. Output
# verified against CPython/PyPy.
N = 40000
BIG = 2**63


def main():
    last = None
    for _ in range(N):
        last = list(enumerate(["a", "b", "c"], BIG))
    print(last)


main()
