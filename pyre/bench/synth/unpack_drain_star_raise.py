# Star-unpack of an iterator of unknown length: `f(*it)` is the consumer that
# routes through `_unpackiterable_unknown_length`, the drain loop jd1
# (`unpackiterable_driver`) traces and compiles. `a, b, c = it` does NOT reach
# it — the exact-arity form has its own `unpack_sequence_exact` loop — so this
# is the shape that exercises the second jit driver end to end.
#
# Three cases, each driven past the jd1 trace threshold:
#   1. plain drain to exhaustion,
#   2. an iterator that raises a non-StopIteration exception mid-drain and
#      re-raises it on every later `next()`,
#   3. the same but NOT exhaustion-stable — it raises once and reports
#      exhausted afterwards. A drain that loses the in-flight exception and
#      lets the interpreter re-derive it by calling `next()` again turns this
#      into a silent early return, so the case pins the exception actually
#      travelling out of the compiled drain / blackhole resume rather than
#      being re-discovered.
N = 4000
ROUNDS = 12


class Counting:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


class Raising:
    """Raises at `at` and keeps raising — exhaustion-stable."""

    def __init__(self, n, at):
        self.i = 0
        self.n = n
        self.at = at

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.at:
            raise ValueError("raising")
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


class RaisingOnce:
    """Raises at `at` exactly once, then reports exhausted."""

    def __init__(self, n, at):
        self.i = 0
        self.n = n
        self.at = at
        self.fired = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.at and not self.fired:
            self.fired = True
            raise ValueError("raising-once")
        if self.fired or self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


def count(*args):
    return len(args)


def main():
    drained = 0
    stable = 0
    once = 0
    for _ in range(ROUNDS):
        drained += count(*Counting(N))
        try:
            count(*Raising(N, N // 2))
        except ValueError as e:
            stable += len(str(e))
        try:
            count(*RaisingOnce(N, N // 2))
        except ValueError as e:
            once += len(str(e))
    print(drained, stable, once)


main()
