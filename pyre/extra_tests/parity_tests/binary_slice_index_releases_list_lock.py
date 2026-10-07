# CPython-suite gap: list BINARY_SLICE holds no per-list lock under the GIL.
# parity-tests reason: a bound `__index__` that waits on another thread's
# list mutation must not deadlock; the stripe covers only the copy.

"""List slice converts bounds before taking the list lock.

`BINARY_SLICE` runs each bound's `__index__` first. An index method that
waits for another thread to mutate this list must return, because that
worker needs the same lock the copy later takes. Holding the stripe
across the conversion deadlocks the starter on the worker and the
worker on the stripe.

The worker is a daemon and a second daemon exits the process if the
start event does not arrive, so a leaked stripe fails this script
instead of hanging the runner.
"""

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


def warmup_list_slice():
    class Idx:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    xs = list(range(8))
    total = 0
    for i in range(N):
        total += len(xs[Idx(1) : Idx(4)])
    return total


def slice_index_waits_for_append():
    started = threading.Event()
    proceed = threading.Event()
    lst = list(range(5))

    class BlockingIdx:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            started.set()
            assert proceed.wait(timeout=5), "mutator did not run"
            return self.value

    def mutator():
        assert started.wait(timeout=5), "slice bound did not start"
        lst.append(99)
        proceed.set()

    worker = threading.Thread(target=mutator, daemon=True)
    worker.start()
    result = lst[BlockingIdx(1) : 4]
    worker.join(timeout=5)
    assert not worker.is_alive(), "mutator still blocked on the list lock"
    assert result == [1, 2, 3]


def watchdog():
    time.sleep(WATCHDOG_S)
    os._exit(2)


threading.Thread(target=watchdog, daemon=True).start()
publish_threads()
assert warmup_list_slice() == 3 * N
slice_index_waits_for_append()
print("OK")
