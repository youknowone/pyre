# pyre-check: gate=1
# ThreadJoinOnShutdown.test_2_join_in_forked_process: the child starts a
# thread that joins the child's main, prints, and the process exits through
# interpreter shutdown (no os._exit).  The child's mutator-registry mutex
# must be OS-backed so unregister at exit does not park into the inherited
# parking_lot HashTable (ubuntu dynasm suite SIGSEGV -11).
import os
import sys
import threading

if not hasattr(os, "fork"):
    print("thread_join_in_forked_process OK")
    raise SystemExit(0)


def joiningfunc(mainthread):
    mainthread.join()
    print("end of thread", flush=True)


childpid = os.fork()
if childpid != 0:
    pid, status = os.waitpid(childpid, 0)
    if status != 0:
        raise AssertionError(f"child status {status}")
    print("thread_join_in_forked_process OK")
    raise SystemExit(0)

t = threading.Thread(target=joiningfunc, args=(threading.current_thread(),))
t.start()
print("end of main", flush=True)
