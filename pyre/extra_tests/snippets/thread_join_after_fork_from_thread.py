# pyre-check: gate=1
# ThreadJoinOnShutdown.test_3_join_in_forked_from_thread: a worker forks,
# and the child joins the pre-fork main Thread.  That handle must already
# be done or join waits forever (ubuntu dynasm suite TIMEOUT 300s).
import os
import sys
import threading

if not hasattr(os, "fork"):
    print("thread_join_after_fork_from_thread OK")
    raise SystemExit(0)

main_thread = threading.current_thread()


def joiningfunc(mainthread):
    mainthread.join()
    print("end of thread", flush=True)


def worker():
    childpid = os.fork()
    if childpid != 0:
        pid, status = os.waitpid(childpid, 0)
        if status != 0:
            raise AssertionError(f"child status {status}")
        return
    t = threading.Thread(target=joiningfunc, args=(main_thread,))
    print("end of main", flush=True)
    if main_thread.is_alive():
        raise AssertionError("pre-fork main Thread still alive in the child")
    t.start()
    t.join()
    os._exit(0)


w = threading.Thread(target=worker)
w.start()
w.join()
print("thread_join_after_fork_from_thread OK")
