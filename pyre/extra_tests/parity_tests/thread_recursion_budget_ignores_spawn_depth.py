# CPython-suite gap: the suite starts its threads from module scope, so nothing
# there measures a worker spawned from inside a deep recursion.
# parity-tests reason: pyre keeps the activation accounting on the
# ExecutionContext, and a worker's context is cloned from the spawning one.  A
# clone that carries the parent's depth over spends part of the worker's
# `sys.setrecursionlimit` budget before the worker runs a single frame, which
# shows up only as recursion stopping early on that thread.

"""A worker thread's recursion budget does not depend on the spawn depth.

The child measures how deep it can recurse under the same limit the main
thread measured, and the two are compared rather than pinned: how many frames
a limit buys differs between implementations, but a thread owes nothing to the
depth its parent happened to be at.
"""

import sys
import threading

LIMIT = 400
SPAWN_DEPTH = 200


def deepest():
    """Recurse until RecursionError and report the depth reached."""
    best = [0]

    def plain(n):
        best[0] = n
        plain(n + 1)

    try:
        plain(0)
    except RecursionError:
        pass
    return best[0]


old_limit = sys.getrecursionlimit()
sys.setrecursionlimit(LIMIT)

result = []
failure = []


def in_thread():
    try:
        result.append(deepest())
    except BaseException as exc:  # surface a worker failure on the main thread
        failure.append(exc)


def spawn_at(depth):
    """Start the worker from `depth` frames down, not from module scope."""
    if depth:
        return spawn_at(depth - 1)
    t = threading.Thread(target=in_thread)
    t.start()
    t.join()
    return None


spawn_at(SPAWN_DEPTH)
assert not failure, failure[0]
child = result[0]
main = deepest()
sys.setrecursionlimit(old_limit)

print("limit          =", LIMIT)
print("spawn depth    =", SPAWN_DEPTH)
print("main depth     =", main)
print("child depth    =", child)
print("lost to parent =", main - child)
# An inherited depth costs the worker about SPAWN_DEPTH of its own budget.
assert main - child < SPAWN_DEPTH // 2, (main, child, SPAWN_DEPTH)
print("OK")
