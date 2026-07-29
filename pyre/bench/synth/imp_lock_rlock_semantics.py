# pyre-check: max-pypy-ratio=23
# Regression guard for the reentrant import lock's Python-visible semantics.
# The lock carries three pieces of state -- the allocated lock, the owning
# execution context, and the recursion depth -- and several of the observables
# below distinguish them, so a collapse back to a bare depth counter shows up
# here rather than only under threads:
#   * lock_held() reports "owned by anyone", not "depth > 0".
#   * an unbalanced release raises RuntimeError, and the message is exact.
#   * the depth is reentrant: N acquires need N releases and lock_held() stays
#     true until the last one.
#   * reinit_lock is interpreter-internal (the fork child hook) and must NOT be
#     reachable from Python.
# Correctness-only fixture: no perf gate header, the work here is trivial.
import _imp


def show(label, fn):
    try:
        fn()
        print(label, "ok")
    except RuntimeError as exc:
        print(label, "RuntimeError:", exc)


show("release-unheld", _imp.release_lock)
print("held-initial:", _imp.lock_held())

_imp.acquire_lock()
print("held-1:", _imp.lock_held())
_imp.acquire_lock()
_imp.acquire_lock()
print("held-3:", _imp.lock_held())
_imp.release_lock()
print("held-2:", _imp.lock_held())
_imp.release_lock()
print("held-1b:", _imp.lock_held())
_imp.release_lock()
print("held-0:", _imp.lock_held())

show("release-overbalanced", _imp.release_lock)
print("reinit_lock exposed:", hasattr(_imp, "reinit_lock"))

_imp.acquire_lock()
_imp.release_lock()
print("held-final:", _imp.lock_held())

import json

print("held-after-import:", _imp.lock_held(), json.dumps([1]))
