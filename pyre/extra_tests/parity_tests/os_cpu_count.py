"""`os.cpu_count()` reports the host's processors, not the interpreter's threads.

The host's processor count is not knowable from inside the test, so what is
asserted is the shape it must have on every host and, above all, the property
that separates a processor count from a thread count: it does not move when
threads come and go.
"""

import os
import posix
import threading


def check(cond, what):
    if not cond:
        raise AssertionError(what)


n = os.cpu_count()
check(n is None or isinstance(n, int), f"cpu_count answered a {type(n).__name__}")
# A bool is an int and would pass the line above while meaning nothing.
check(not isinstance(n, bool), "cpu_count answered a bool")
if n is not None:
    check(n > 0, f"cpu_count answered {n}")

# The private alias is reached through `posix`: `os` star-imports, which skips
# every underscore name, so `os._cpu_count` does not exist on either side.
check(not hasattr(os, "_cpu_count"), "os grew a _cpu_count")
if hasattr(posix, "_cpu_count"):
    check(posix._cpu_count() == n, f"_cpu_count {posix._cpu_count()} disagrees with cpu_count {n}")

# ── the property ─────────────────────────────────────────────────────────
# A count wired to the process's own thread table — /proc/self/stat's
# num_threads, or the mach task_threads count — rises here and falls again.
# A processor count is the same number all three times.
started = threading.Semaphore(0)
release = threading.Event()


def hold():
    started.release()
    release.wait()


before = os.cpu_count()
threads = [threading.Thread(target=hold) for _ in range(6)]
for t in threads:
    t.start()
try:
    for _ in threads:
        started.acquire()
    # All six are alive and parked at this point.
    during = os.cpu_count()
finally:
    release.set()
    for t in threads:
        t.join()
after = os.cpu_count()

check(
    before == during == after,
    f"cpu_count moved with the live thread count: {before} -> {during} -> {after}",
)

# ── the value, not just its stability ────────────────────────────────────
# A constant wrong answer would satisfy everything above. The processor count
# the host reports through sysconf is the one both sides are built on — the
# `sysconf(_SC_NPROCESSORS_ONLN)` arm directly, the `sysctl(CTL_HW, HW_NCPU)`
# arm because a host reports the same processors either way.
if before is not None and hasattr(os, "sysconf"):
    try:
        onln = os.sysconf("SC_NPROCESSORS_ONLN")
    except (ValueError, OSError):
        onln = None
    if onln is not None and onln > 0:
        check(before == onln, f"cpu_count {before} is not the host's {onln} processors")

# ── consistency with the neighbouring counts ─────────────────────────────
# The affinity mask is a subset of the processors, so it can never be wider.
if hasattr(os, "sched_getaffinity") and before is not None:
    mask = len(os.sched_getaffinity(0))
    check(before >= mask, f"cpu_count {before} is narrower than the affinity mask {mask}")

# process_cpu_count is either the mask or cpu_count itself; neither exceeds it.
if hasattr(os, "process_cpu_count") and before is not None:
    p = os.process_cpu_count()
    check(p is None or p <= before, f"process_cpu_count {p} exceeds cpu_count {before}")

print("OK")
