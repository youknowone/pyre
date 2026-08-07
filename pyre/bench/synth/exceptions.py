# pyre-check: max-pypy-ratio=118
# pyre-check: skip-cpython
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
# 41258646 puts pypy at 63ms, barely clear of it; every count that leaves pypy
# measurable runs cpython past its reference timeout, since cpython raises and
# catches all 41M of these for real. cpython is dropped deliberately rather
# than by spending the whole timeout to discover the same drop.
N = 41258646


def may_fail(i):
    if (i & 31) == 0:
        raise ValueError(i)
    return i & 7


def main():
    i = 0
    acc = 0
    while i < N:
        try:
            acc = acc + may_fail(i)
        except ValueError as e:
            acc = acc - (e.args[0] & 15)
        finally:
            acc = acc + 1
        i = i + 1
    print(acc)


main()

