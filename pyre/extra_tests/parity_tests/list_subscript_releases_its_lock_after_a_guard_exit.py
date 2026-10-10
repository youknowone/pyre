# CPython-suite gap: `test_list` and `test_free_threading/test_list` subscript
# lists from several threads, but none of them first compiles a subscript loop
# and then changes the storage kind of the list it reads.
#
# parity-tests reason: once a second thread has started, `list[i]` takes the
# list's lock around the read. A compiled subscript whose storage guard fails
# between the acquire and the release has to leave with the lock released, or
# the next thread to touch any list waits forever.
#
# CPython 3.14 and PyPy agree on every line below.
import threading


def another_thread_can_use_a_list():
    done = []

    def worker():
        items = [1]
        items.append(2)
        done.append(items[0])

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(3)
    return bool(done)


def load(items, i):
    return items[i]


ints = [0] * 8
objs = [None] * 8
flts = [0.0] * 8

# The lock is only taken after a thread has existed.
assert another_thread_can_use_a_list()
for rep in range(6):
    for i in range(3000):
        load(ints, -1 - (i & 7))
    # Other storage kinds fail the guards inside the compiled read.
    for i in range(3000):
        load(objs if i & 1 else flts, -1 - (i & 7))
        load(ints, i & 7)
    assert another_thread_can_use_a_list(), rep
print("OK")
