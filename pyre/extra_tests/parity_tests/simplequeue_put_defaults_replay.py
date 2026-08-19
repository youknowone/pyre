# CPython-suite gap: test_queue reaches this only through
# `CSimpleQueueTest.test_reentrancy`, which reads as a finalizer-ordering test
# and reports it as a 10000-element list diff -- it names neither the replayed
# call nor the argument shape that causes it.
# parity-tests reason: the defect is a miscompile, so it needs a fixture that
# runs the loop hot on both backends and checks a count that the interpreter
# and the compiled trace must agree on.

"""A hot loop must call its producer exactly once per iteration.

`SimpleQueue.put` is declared `put(item, block=True, timeout=None)`; the two
trailing arguments are accepted and ignored, because an unbounded queue never
blocks. Writing `put(v)` therefore leaves the call site to fill both defaults
in, and that is the shape this guards.

When such a call is compiled into a hot loop, the call feeding it must not be
re-executed. `Counter.next` is deliberately side-effecting, so a replay shows
up twice over: as a producer count that exceeds the iteration count, and as a
value that is generated but never queued -- which shifts every later result by
one rather than reordering a pair.

A plain function call in the producer position does not reproduce it; the call
has to go through the method path, which is why this uses a bound method.

The count is small enough to stay fast and large enough for the loop to be
compiled and left at least once.
"""

import queue

LIMIT = 1500


class Counter:
    def __init__(self):
        self.n = 0

    def next(self):
        value = self.n
        self.n += 1
        return value


def main():
    q = queue.SimpleQueue()
    counter = Counter()
    results = []

    while True:
        q.put(counter.next())
        results.append(q.get())
        if results[-1] >= LIMIT:
            break

    assert counter.n == len(results), (
        "the producer ran more often than the loop body",
        counter.n,
        len(results),
    )
    expected = list(range(LIMIT + 1))
    assert results == expected, (
        "a produced value never reached the queue",
        next(
            (i, results[i], expected[i])
            for i in range(len(results))
            if results[i] != expected[i]
        ),
    )

    print("OK")


main()
