# CPython-suite gap: test_itertools.TestBasicOps.test_tee_concurrent deadlocks
# under the JIT after earlier TestBasicOps list iteration.
# parity-tests reason: a compiled list FOR_ITER must not hold the list class
# stripe across the step; a later thread that unpacks a list stararg waits on
# that stripe, and the starter waits on the worker.

"""A locked list-iterator step must release before another thread getitems.

Once any thread has started, list iteration takes the stripe-locked arm.
A hot `for` over a list compiles that arm. The acquire and release of the
stripe have to finish inside the same residual: if a compiled step acquires
and a later guard or resume leaves the stripe held, `Thread(target=f,
args=[x])` cannot unpack `args` and the starter waiting on that worker
never observes the start event.

The worker is a daemon and a second daemon exits the process if the start
event does not arrive, so a leaked stripe fails this script instead of
hanging the runner.
"""

import itertools
import os
import threading
import time

N = 4000
WATCHDOG_S = 12


def publish_threads():
    done = threading.Event()

    def mark():
        done.set()

    thread = threading.Thread(target=mark)
    thread.start()
    thread.join()


def warmup_list_iter():
    xs = list(range(N))
    total = 0
    for value in xs:
        total += value
    return total


def tee_concurrent():
    start = threading.Event()
    finish = threading.Event()

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            start.set()
            finish.wait()

    left, right = itertools.tee(Source())
    worker = threading.Thread(target=next, args=[left], daemon=True)
    worker.start()
    try:
        assert start.wait(timeout=5), "stararg worker did not start"
        raised = False
        try:
            next(right)
        except RuntimeError as exc:
            raised = "tee" in str(exc)
        assert raised, "expected RuntimeError mentioning tee"
    finally:
        finish.set()
        worker.join(timeout=5)
        assert not worker.is_alive(), "stararg worker still blocked"


def watchdog():
    time.sleep(WATCHDOG_S)
    os._exit(2)


threading.Thread(target=watchdog, daemon=True).start()
publish_threads()
assert warmup_list_iter() == N * (N - 1) // 2
tee_concurrent()
print("OK")
