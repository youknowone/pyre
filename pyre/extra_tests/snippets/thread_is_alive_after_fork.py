# pyre-check: gate=1
# ThreadTests.test_is_alive_after_fork: a worker that has already
# finished is gone from ACTIVE_HANDLES.  The child's is_alive() must
# still see is_done() without waiting on the inherited handle mutex
# (ubuntu dynasm suite hang when that mutex is a parking_lot waiter).
import os
import sys
import threading
import warnings

if not hasattr(os, "fork"):
    print("thread_is_alive_after_fork OK")
    raise SystemExit(0)

failures = []
for i in range(40):
    t = threading.Thread(target=lambda: None)
    t.start()
    # Both sides of the race: fork while the worker is still running,
    # and fork after it has already finished and left ACTIVE_HANDLES.
    if i % 2 == 0:
        t.join()
    with warnings.catch_warnings(category=DeprecationWarning, action="ignore"):
        child = os.fork()
    if child == 0:
        alive = t.is_alive()
        os._exit(11 if alive else 10)
    t.join()
    pid, status = os.waitpid(child, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 10:
        continue
    failures.append((i, status))

if failures:
    raise AssertionError(f"child still saw is_alive: {failures}")
print("thread_is_alive_after_fork OK")
