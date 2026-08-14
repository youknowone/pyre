# CPython-suite gap: recursion tests do not compare cold and JIT-hot call depth.
# parity-tests reason: this guards pyre's native budget across JIT guard resume.

"""`sys.setrecursionlimit` must keep deciding how deep recursion goes after
the recursive function is JIT-hot.

The recursion depth is bounded twice: by the logical limit, and by a native
byte budget (`stack_check.rs` `MAX_STACK_SIZE`, scaled by `recursionlimit /
1000`). The budget is meant to sit above the limit's own cutoff so the limit is
what a program observes. A level that runs as compiled code and leaves it
through a guard failure nests the whole resume chain on the native stack and
costs several times what an interpreted level costs, so a budget calibrated
against the interpreter alone drops below the limit as soon as the function
goes hot: the same function measured five times in a row returned 997, 997,
997, 997, then 458.

Reading the same depth every round is the property; the absolute number is not
asserted, only that it is the limit and not the guard that stopped the
recursion.
"""

import sys
import threading


def deepest():
    """Recurse until RecursionError and report the depth reached."""
    best = [0]

    def plain(n) -> None:
        best[0] = n
        plain(n + 1)

    try:
        plain(0)
    except RecursionError:
        pass
    return best[0]


def readings(rounds=12):
    # A fresh code object per call would warm up separately; the point is that
    # one code object stays stable across repeats, so `deepest` is shared.
    return [deepest() for _ in range(rounds)]


def check(where, values, limit):
    assert len(set(values)) == 1, (where, "depth moved across repeats", values)
    reached = values[0]
    assert reached <= limit, (where, "deeper than the limit", reached, limit)
    assert reached >= limit * 9 // 10, (
        where,
        "stopped well short of the limit, so the byte guard fired first",
        reached,
        limit,
    )


limit = sys.getrecursionlimit()
check("default limit", readings(), limit)

# A thread gets its own native stack, and its own chance to be sized too small
# for the limit it is handed.
failure = []


def in_thread():
    try:
        check("thread", readings(), sys.getrecursionlimit())
    except BaseException as exc:  # surface every worker failure on the main thread
        failure.append(exc)


t = threading.Thread(target=in_thread)
t.start()
t.join()
assert not failure, failure[0]

sys.setrecursionlimit(2000)
check("raised limit", readings(rounds=6), 2000)
sys.setrecursionlimit(limit)

print("OK")
