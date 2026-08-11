# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
# The wasm guest has no OS-thread implementation, while this fixture verifies
# which native mutator dispatches an object-space GC action.
import _thread
import gc


main_ident = _thread.get_ident()
callback_idents = []
worker_idents = []
done = _thread.allocate_lock()
done.acquire()


def on_collect(stats):
    callback_idents.append(_thread.get_ident())


def worker():
    worker_idents.append(_thread.get_ident())
    gc.collect()
    done.release()


gc.hooks.on_gc_collect = on_collect
_thread.start_new_thread(worker, ())
done.acquire()

assert callback_idents
assert callback_idents[-1] == worker_idents[-1]
assert callback_idents[-1] != main_ident
print("gc hook ran on collecting worker")
