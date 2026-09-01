# pyre-check: max-pypy-ratio=5.3
# A `with` block whose context manager is allocated inside the loop.  Nothing
# it builds survives the iteration, so the instance, the two bound methods
# LOAD_SPECIAL resolves, and the conditionally-bound `obj` the trailing read
# needs are all shapes the tracer has to see through before the body can come
# out as plain integer arithmetic.
# N is sized so pypy's own execution stays well clear of the ratio gate's
# minimum baseline: under that bar the gate divides by the measurement floor
# and the printed ratio reads process startup rather than this loop.
N = 150000000


class Accumulator:
    def __init__(self, x):
        self.x = x

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.x = self.x + 1
        return False


def main():
    i = 0
    acc = 0
    while i < N:
        with Accumulator(i & 15) as obj:
            acc = acc + obj.x
        acc = acc + obj.x
        i = i + 1
    print(acc)


main()
