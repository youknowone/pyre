"""os.lockf, os.waitid and the two sparse-file SEEK_* values are real.

None of the three existed. They are POSIX rather than Linux-only — the Apple
targets carry all of them — so what is checked here is the behaviour, not just
the presence: a lock that another descriptor can see, a child reported by
waitid without being reaped, and whence values distinct from the three os.py
fixes itself.
"""

import os
import sys
import tempfile

import atexit
import shutil


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(call, exc):
    try:
        call()
    except exc as e:
        return e
    raise AssertionError(f"{exc.__name__} was not raised")


if sys.platform == "win32":
    for name in ("lockf", "F_LOCK", "waitid", "waitid_result", "SEEK_HOLE"):
        check(not hasattr(os, name), f"windows grew an os.{name}")
    print("OK")
    raise SystemExit

d = tempfile.mkdtemp()
atexit.register(shutil.rmtree, d, ignore_errors=True)
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"0123456789" * 10)

# ── lockf ─────────────────────────────────────────────────────────────────
# The four commands are one vocabulary, so they are four different numbers.
cmds = (os.F_ULOCK, os.F_LOCK, os.F_TLOCK, os.F_TEST)
check(len(set(cmds)) == 4, f"the lockf commands are not distinct: {cmds}")

fd = os.open(p, os.O_RDWR)
try:
    # A whole-file exclusive lock, then the same descriptor tests it — which
    # succeeds, because a process never contends with itself.
    check(os.lockf(fd, os.F_LOCK, 0) is None, "lockf answered something")
    check(os.lockf(fd, os.F_TEST, 0) is None, "F_TEST on our own lock")
    os.lockf(fd, os.F_ULOCK, 0)

    # A locked region is a region: seek first, lock a length, and the lock
    # covers from there.
    os.lseek(fd, 10, os.SEEK_SET)
    os.lockf(fd, os.F_LOCK, 5)
    os.lockf(fd, os.F_ULOCK, 5)
    os.lseek(fd, 0, os.SEEK_SET)

    raises(lambda: os.lockf(fd, -12345, 0), OSError)
finally:
    os.close(fd)

raises(lambda: os.lockf(-1, os.F_TEST, 0), OSError)

# ── the sparse-file whence values ─────────────────────────────────────────
whence = (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END, os.SEEK_HOLE, os.SEEK_DATA)
check(len(set(whence)) == 5, f"the whence values are not distinct: {whence}")
check((os.SEEK_SET, os.SEEK_CUR, os.SEEK_END) == (0, 1, 2), "os.py's three moved")

fd = os.open(p, os.O_RDONLY)
try:
    # Every byte of this file was written, so the data starts at 0 and the
    # only hole is the implicit one at the end.
    check(os.lseek(fd, 0, os.SEEK_DATA) == 0, "SEEK_DATA did not find the data at 0")
    check(os.lseek(fd, 0, os.SEEK_HOLE) == 100, "SEEK_HOLE is not the end of the file")
finally:
    os.close(fd)

# ── waitid ────────────────────────────────────────────────────────────────
ids = (os.P_ALL, os.P_PID, os.P_PGID)
check(len(set(ids)) == 3, f"the waitid id types are not distinct: {ids}")
codes = (
    os.CLD_EXITED,
    os.CLD_KILLED,
    os.CLD_DUMPED,
    os.CLD_TRAPPED,
    os.CLD_STOPPED,
    os.CLD_CONTINUED,
)
check(len(set(codes)) == 6, f"the CLD_* codes are not distinct: {codes}")
opts = (os.WEXITED, os.WSTOPPED, os.WNOWAIT)
check(len(set(opts)) == 3, f"the waitid options are not distinct: {opts}")
check(all(o != 0 and o & (o - 1) == 0 for o in opts), f"a waitid option is not a bit: {opts}")

pid = os.fork()
if pid == 0:
    os._exit(7)

# WNOWAIT reports without reaping, so the same child is still there to be
# waited for afterwards — which is the whole reason the call exists.
info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
check(info is not None, "waitid answered None for a child that exited")
check(isinstance(info, tuple), f"waitid_result is not a tuple: {type(info).__name__}")
check(len(info) == 5, f"waitid_result has {len(info)} fields")
check(info.si_pid == pid, f"si_pid {info.si_pid} is not the child {pid}")
# si_uid is the child's real uid, which Darwin leaves at 0 rather than filling
# in — so the field is checked for being read out at all, not for its value.
check(isinstance(info.si_uid, int), f"si_uid {info.si_uid!r}")
check(info.si_status == 7, f"si_status {info.si_status} is not the exit code")
check(info.si_code == os.CLD_EXITED, f"si_code {info.si_code} is not CLD_EXITED")
check(info[0] == info.si_pid, "the tuple body and the fields disagree")

# Still unreaped, so this one both reports and reaps it.
again = os.waitid(os.P_PID, pid, os.WEXITED)
check(again.si_pid == pid, "the second waitid lost the child")

# And now there is nothing left to reap.
raises(lambda: os.waitid(os.P_PID, pid, os.WEXITED), ChildProcessError)

print("OK")
