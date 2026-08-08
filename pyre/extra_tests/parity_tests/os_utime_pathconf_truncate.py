"""Three answers that were the wrong value rather than the wrong call.

- utime carried its timestamps as an unsigned duration, so every time before
  the epoch — which POSIX writes and CPython accepts — was refused instead.
- pathconf answered None where the host has no determinate limit; the number
  the call returns for that is -1, and None is neither that value nor a type a
  caller can compare against a limit.
- truncate narrowed its length to off_t with a cast, so a value too large to be
  a size became a different size rather than an error.
"""

import atexit
import os
import shutil
import sys
import tempfile


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def raises(call, exc):
    try:
        call()
    except exc as e:
        return e
    raise AssertionError(f"{exc.__name__} was not raised")


d = tempfile.mkdtemp()
atexit.register(shutil.rmtree, d, ignore_errors=True)
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"0123456789")

# ── utime before the epoch ────────────────────────────────────────────────
# The seconds are the floor of the value and the nanoseconds are what is left
# above it, so -1ns is the last nanosecond of 1969 rather than a value with no
# representation.
#
# Windows is left out: a FILETIME counts 100ns ticks, so a nanosecond that is
# not a multiple of 100 is not a time that filesystem can hold — CPython reads
# -1 back as -100 — and this build's Windows path carries the timestamp as an
# unsigned duration and refuses the whole range. See the follow-up task.
if sys.platform != "win32":
    os.utime(p, ns=(-1, -1))
    check(os.stat(p).st_mtime_ns == -1, f"utime(ns=(-1,-1)) -> {os.stat(p).st_mtime_ns}")
    os.utime(p, ns=(-2_500_000_000, -2_500_000_000))
    check(os.stat(p).st_mtime_ns == -2_500_000_000, "utime(ns) lost a negative second")

    os.utime(p, (-1.5, -2.5))
    check(
        os.stat(p).st_mtime_ns == -2_500_000_000,
        f"utime((-1.5,-2.5)) -> {os.stat(p).st_mtime_ns}",
    )
    check(os.stat(p).st_atime_ns == -1_500_000_000, "utime(times) lost the access time")

    # The same through a descriptor, which is the form supports_fd advertises.
    if os.utime in os.supports_fd:
        fd = os.open(p, os.O_RDWR)
        try:
            os.utime(fd, ns=(-3_000_000_000, -4_000_000_000))
            check(
                os.stat(p).st_mtime_ns == -4_000_000_000,
                "utime(fd, ns) lost a negative second",
            )
        finally:
            os.close(fd)

# `times` is the one argument here that may be spelled either way — it sits
# before the keyword-only marker. The pair is one the platform holds: Windows
# refuses times before the epoch, for the reason given above.
atime, mtime = (5.0, 6.0) if sys.platform == "win32" else (-5.0, -6.0)
os.utime(p, times=(atime, mtime))
check(
    os.stat(p).st_mtime_ns == int(mtime) * 1_000_000_000,
    f"utime(times=...) by keyword -> {os.stat(p).st_mtime_ns}",
)
raises(lambda: os.utime(p, (1, 2), times=(3, 4)), TypeError)

# Back to a time the rest of the file can be reasoned about.
os.utime(p, ns=(1_000_000_000, 2_000_000_000))

# ── pathconf's indeterminate limit ────────────────────────────────────────
# Every name the table carries either answers with a number or refuses the
# question — the terminal-only limits are not ones a regular file has. What no
# answer may be is None: a host with no determinate value says so with -1.
#
# `pathconf` and the `pathconf_names` table it resolves through are a POSIX
# surface; neither runtime carries them on Windows, so there is nothing to
# compare there.
def limits(target):
    for name in sorted(os.pathconf_names):
        try:
            limit = os.pathconf(target, name)
        except OSError:
            continue
        check(isinstance(limit, int), f"pathconf({name!r}) answered {limit!r}")
        check(limit >= -1, f"pathconf({name!r}) answered {limit}")
        yield name, limit


if sys.platform != "win32":
    answered = dict(limits(p))
    check(answered, "pathconf answered no name at all")
    check("PC_NAME_MAX" in answered, "pathconf refused PC_NAME_MAX on a regular file")

    if os.pathconf in os.supports_fd:
        fd = os.open(p, os.O_RDONLY)
        try:
            by_fd = dict(limits(fd))
            check(
                by_fd.get("PC_NAME_MAX") == answered["PC_NAME_MAX"],
                "the descriptor and the name disagree about PC_NAME_MAX",
            )
        finally:
            os.close(fd)

    raises(lambda: os.pathconf(p, "PC_NOT_A_REAL_NAME"), ValueError)

# ── truncate's length ─────────────────────────────────────────────────────
# A length wider than off_t is not a size the file can be given; the cast that
# used to narrow it would have picked a different one.
raises(lambda: os.truncate(p, 2**64), OverflowError)
raises(lambda: os.truncate(p, -(2**64)), OverflowError)
check(os.stat(p).st_size == 10, "the refused truncate changed the file")

fd = os.open(p, os.O_RDWR)
try:
    raises(lambda: os.ftruncate(fd, 2**64), OverflowError)
    check(os.stat(p).st_size == 10, "the refused ftruncate changed the file")
    os.ftruncate(fd, 4)
finally:
    os.close(fd)
check(os.stat(p).st_size == 4, "ftruncate did not shorten the file")

os.truncate(p, 2)
check(os.stat(p).st_size == 2, "truncate did not shorten the file")

if sys.platform != "win32":
    # `truncate` on a name it cannot open reports the name, not a bare errno.
    e = raises(lambda: os.truncate(os.path.join(d, "absent"), 0), OSError)
    check(e.filename == os.path.join(d, "absent"), f"truncate lost the filename: {e.filename!r}")

print("OK")
