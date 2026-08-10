"""The Linux-only posix names: pipe2 and the scheduling-policy group.

`pipe2` and `sched_getparam`/`sched_setparam`/`sched_getscheduler`/
`sched_setscheduler` were absent, and so was the `sched_param` type the four
exchange. The group is the host's, not the module's — a host whose libc has no
`sched_setscheduler` publishes none of it — so what is asserted here is the
shape that holds wherever the names do exist, plus the fact that they exist
nowhere else.

`os.dup3` is deliberately not checked for: no host publishes such a name.
"""

import os
import sys


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(call, exc):
    try:
        call()
    except exc:
        return
    raise AssertionError(f"{exc.__name__} was not raised")


check(not hasattr(os, "dup3"), "os grew a dup3")

# ── the <fcntl.h> flag set ───────────────────────────────────────────────
# The values are the host header's own, so what is checked is which names
# exist and that each is an int. The split is the hosts' own: seven on every
# Unix, then one group per platform. `nt` has none of them.
FLAGS = {
    "": ("O_ACCMODE", "O_ASYNC", "O_CLOEXEC", "O_DIRECTORY", "O_FSYNC", "O_NOCTTY",
         "O_NOFOLLOW"),
    "linux": ("O_DIRECT", "O_LARGEFILE", "O_NOATIME", "O_PATH", "O_RSYNC", "O_TMPFILE"),
    "darwin": ("O_EVTONLY", "O_EXEC", "O_EXLOCK", "O_NOFOLLOW_ANY", "O_SEARCH", "O_SHLOCK",
               "O_SYMLINK"),
}
if sys.platform != "win32":
    expected = FLAGS[""]
    for key, names in FLAGS.items():
        if key and sys.platform.startswith(key):
            expected += names
    for name in expected:
        check(hasattr(os, name), f"os has no {name}")
        check(isinstance(getattr(os, name), int), f"{name} is not an int")
    # O_CLOEXEC is the flag `pipe2` exists for, and a zero there would be a flag
    # that silently leaves the descriptor inheritable.
    check(os.O_CLOEXEC != 0, "O_CLOEXEC is 0")

# ── pipe2 ────────────────────────────────────────────────────────────────
if hasattr(os, "pipe2"):
    # Unlike `pipe`, no inheritance is forced on the pair: the flags argument is
    # the whole of the caller's control over it.
    r, w = os.pipe2(0)
    try:
        check(isinstance(r, int) and isinstance(w, int), "pipe2 answered non-ints")
        check(r != w, "pipe2 answered one descriptor twice")
        check(os.get_inheritable(r), "pipe2(0) read end is not inheritable")
        check(os.get_inheritable(w), "pipe2(0) write end is not inheritable")
        os.write(w, b"x")
        check(os.read(r, 1) == b"x", "the pipe2 pair is not connected")
    finally:
        os.close(r)
        os.close(w)

    r, w = os.pipe2(os.O_CLOEXEC)
    try:
        check(not os.get_inheritable(r), "pipe2(O_CLOEXEC) read end is inheritable")
        check(not os.get_inheritable(w), "pipe2(O_CLOEXEC) write end is inheritable")
    finally:
        os.close(r)
        os.close(w)

    raises(lambda: os.pipe2(), TypeError)
    raises(lambda: os.pipe2("x"), TypeError)
    raises(lambda: os.pipe2(-1), OSError)
elif sys.platform.startswith(("linux", "freebsd", "netbsd", "openbsd", "dragonfly")):
    raise AssertionError("this host has pipe2 and os does not publish it")

# ── the CPU affinity mask ────────────────────────────────────────────────
if hasattr(os, "sched_getaffinity"):
    cpus = os.sched_getaffinity(0)
    check(isinstance(cpus, set), f"sched_getaffinity answered a {type(cpus).__name__}")
    check(cpus, "this process is affine to no CPU at all")
    check(all(isinstance(c, int) for c in cpus), "a CPU number is not an int")
    check(all(c >= 0 for c in cpus), "a CPU number is negative")

    raises(lambda: os.sched_getaffinity(), TypeError)
    raises(lambda: os.sched_getaffinity("x"), TypeError)
    # A pid nobody runs under. ProcessLookupError is an OSError.
    raises(lambda: os.sched_getaffinity(-1), OSError)

    try:
        # The mask this process already has, spelled three ways: a set, a list
        # and a bare iterator are all accepted, and all answer None.
        check(os.sched_setaffinity(0, cpus) is None, "sched_setaffinity answered a value")
        check(os.sched_setaffinity(0, list(cpus)) is None, "a list mask was refused")
        check(os.sched_setaffinity(0, iter(cpus)) is None, "an iterator mask was refused")
        check(os.sched_getaffinity(0) == cpus, "the mask changed under a no-op write")
    except PermissionError:
        # A host that refuses a scheduler write refuses it here too.
        pass

    # An empty mask leaves nothing to run on, and the kernel says so.
    raises(lambda: os.sched_setaffinity(0, []), OSError)
    # A CPU that cannot be one: negative, wider than the mask, wider than a C int.
    raises(lambda: os.sched_setaffinity(0, [-1]), ValueError)
    raises(lambda: os.sched_setaffinity(0, [99999]), OSError)
    raises(lambda: os.sched_setaffinity(0, [2**40]), OverflowError)
    # Not a CPU number at all, and not an iterable at all.
    raises(lambda: os.sched_setaffinity(0, ["x"]), TypeError)
    raises(lambda: os.sched_setaffinity(0, 5), TypeError)
    raises(lambda: os.sched_setaffinity(0), TypeError)
    raises(lambda: os.sched_setaffinity(-1, cpus), OSError)

    try:
        os.sched_setaffinity(0, cpus)
    except PermissionError:
        pass
    check(os.sched_getaffinity(0) == cpus, "the mask did not survive the refusals")

# ── sched_param and the policy calls ─────────────────────────────────────
if not hasattr(os, "sched_getparam"):
    check(not hasattr(os, "sched_param"), "sched_param without sched_getparam")
    check(not hasattr(os, "sched_setparam"), "sched_setparam without sched_getparam")
    check(not hasattr(os, "sched_getscheduler"), "sched_getscheduler alone")
    print("OK")
    raise SystemExit

# The type carries one field and takes the priority itself, not a sequence.
param = os.sched_param(5)
check(type(param).__name__ == "sched_param", f"the type is {type(param).__name__}")
check(os.sched_param.__module__ == "posix", f"module is {os.sched_param.__module__!r}")
check(os.sched_param.n_fields == 1, f"n_fields is {os.sched_param.n_fields}")
check(isinstance(param, tuple), "sched_param is not a tuple")
check(len(param) == 1, f"len(sched_param(5)) is {len(param)}")
check(param[0] == 5, f"sched_param(5)[0] is {param[0]!r}")
check(param.sched_priority == 5, f"sched_priority is {param.sched_priority!r}")
check(repr(param) == "posix.sched_param(sched_priority=5)", f"repr is {repr(param)}")
check(os.sched_param(0) == os.sched_param(0), "two equal sched_params differ")

# Reading this process's own policy and priority always works.
policy = os.sched_getscheduler(0)
check(isinstance(policy, int), f"sched_getscheduler answered {policy!r}")
# SCHED_BATCH and SCHED_IDLE are Linux's alone, so the set is the host's.
known = [getattr(os, n) for n in ("SCHED_OTHER", "SCHED_FIFO", "SCHED_RR",
                                  "SCHED_BATCH", "SCHED_IDLE") if hasattr(os, n)]
check(policy in known, f"sched_getscheduler answered an unknown policy {policy!r}")
current = os.sched_getparam(0)
check(isinstance(current, os.sched_param), f"sched_getparam answered {type(current).__name__}")
check(isinstance(current.sched_priority, int), "sched_priority is not an int")

raises(lambda: os.sched_getparam(), TypeError)
raises(lambda: os.sched_getscheduler("x"), TypeError)
# A pid nobody runs under, and one that is not a pid at all.
raises(lambda: os.sched_getparam(-1), OSError)
raises(lambda: os.sched_getscheduler(-1), OSError)

# The round-robin quantum is one float, not the timespec pair the call fills.
if hasattr(os, "sched_rr_get_interval"):
    quantum = os.sched_rr_get_interval(0)
    check(isinstance(quantum, float), f"sched_rr_get_interval answered {quantum!r}")
    check(quantum >= 0.0, f"sched_rr_get_interval answered {quantum!r}")
    raises(lambda: os.sched_rr_get_interval(), TypeError)
    raises(lambda: os.sched_rr_get_interval("x"), TypeError)
    raises(lambda: os.sched_rr_get_interval(-1), OSError)

if not hasattr(os, "sched_setparam"):
    print("OK")
    raise SystemExit

# Both setters answer None, and both demand the type rather than a bare int or
# a tuple that would index the same.
try:
    check(os.sched_setparam(0, current) is None, "sched_setparam answered a value")
    check(os.sched_setscheduler(0, policy, current) is None,
          "sched_setscheduler answered a value")
except PermissionError:
    # A host that refuses a scheduler write refuses it here too, and nothing
    # below depends on the write having gone through.
    pass
raises(lambda: os.sched_setparam(0, 5), TypeError)
raises(lambda: os.sched_setparam(0, (5,)), TypeError)
raises(lambda: os.sched_setscheduler(0, policy, 5), TypeError)
raises(lambda: os.sched_setparam(0), TypeError)

# A priority the C int cannot hold is refused before the call is made.
raises(lambda: os.sched_setparam(0, os.sched_param(2**31)), OverflowError)
raises(lambda: os.sched_setparam(0, os.sched_param(-(2**31) - 1)), OverflowError)
raises(lambda: os.sched_setscheduler(0, policy, os.sched_param(2**31)), OverflowError)

# A policy the host does not define, and a priority outside the policy's band,
# are the host's to refuse.
raises(lambda: os.sched_setscheduler(0, 12345, current), OSError)
raises(lambda: os.sched_setparam(0, os.sched_param(99)), OSError)

print("OK")
