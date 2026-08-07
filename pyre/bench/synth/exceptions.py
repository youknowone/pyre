# pyre-check: max-pypy-ratio=118
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
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

