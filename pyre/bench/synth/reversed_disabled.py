# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=1` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# reversed() treats `__reversed__ = None` as not reversible (even with a
# sequence protocol) and propagates a raised __reversed__ instead of yielding a
# corrupt null object. Output verified against CPython/PyPy.
N = 40000


class Disabled:
    __reversed__ = None

    def __getitem__(self, i):
        return i

    def __len__(self):
        return 3


class Custom:
    def __reversed__(self):
        return iter([9, 8, 7])


def main():
    disabled = 0
    total = 0
    for i in range(N):
        try:
            reversed(Disabled())
        except TypeError:
            disabled += 1
        total += sum(reversed(Custom()))
        total += sum(reversed([1, 2, 3]))
    print(disabled, total)


main()
