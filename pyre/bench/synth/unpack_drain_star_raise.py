# pyre-check: max-pypy-ratio=106
# The ceiling is twice the slowest ratio observed (52.7x on the linux runner),
# rounded up.
# Star-unpack of an iterator of unknown length: `f(*it)` is the consumer that
# routes through `_unpackiterable_unknown_length`, the drain loop jd1
# (`unpackiterable_driver`) traces and compiles. `a, b, c = it` does NOT reach
# it — the exact-arity form has its own `unpack_sequence_exact` loop — so this
# is the shape that exercises the second jit driver end to end.
#
# Five cases, each driven past the jd1 trace threshold:
#   1. plain drain to exhaustion,
#   2. an iterator that raises a non-StopIteration exception mid-drain and
#      re-raises it on every later `next()`,
#   3. the same but NOT exhaustion-stable — it raises once and reports
#      exhausted afterwards. A drain that loses the in-flight exception and
#      lets the interpreter re-derive it by calling `next()` again turns this
#      into a silent early return, so the case pins the exception actually
#      travelling out of the compiled drain / blackhole resume rather than
#      being re-discovered.
#
# Cases 1-3 all raise long after the drain has been compiled, so they only
# exercise the live-enter/blackhole exits. The two below raise *while the
# trace is being recorded*, which is a different exit (the walk drains its
# framestack and returns an exception-carrying finish):
#   4. raises at EARLY_AT, chosen to land inside the very first traced drain,
#   5. `exc.args = <list>` from inside a hot loop — an unknown-length drain
#      that ends on its own loop-exit StopIteration while being traced. That
#      StopIteration is the loop's normal terminator and must not reach the
#      interpreter. The store is off the hot path (`i % ARGS_EVERY`) so the
#      enclosing loop compiles without it and the drain is traced from a guard
#      exit, which is where the leak shows.
N = 4000
ROUNDS = 12
# One less than the jd1 trace threshold, so the raise happens on the crossing
# that starts the trace rather than after the loop is already compiled.
EARLY_AT = 99
ARGS_ITERS = 40000
ARGS_EVERY = 997


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


class RaisingEarly:
    """Raises once at `at`, then reports exhausted — same shape as
    `RaisingOnce` but on its own type so it gets its own jd1 green key and
    its raise lands on the trace-recording crossing."""

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
            raise ValueError("raising-early")
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
    early = 0
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
        try:
            count(*RaisingEarly(N, EARLY_AT))
        except ValueError as e:
            early += len(str(e))

    print(drained, stable, once, early)


def args_drain():
    """`exc.args = <iterable>` is the other unknown-length drain consumer
    (`coerce_to_list_for_args`). The list is short, so the drain always ends on
    its own StopIteration — including on the crossing that records the trace."""
    exc = ValueError("x")
    items = [1, 2, 3]
    n = 0
    for i in range(ARGS_ITERS):
        if i % ARGS_EVERY == ARGS_EVERY - 1:
            exc.args = items
        n += 1
    print(n, exc.args)


# `args_drain` runs first: measured, the StopIteration leak is only
# observable on a cold JIT, so reordering these two hides case 5.
args_drain()
main()
