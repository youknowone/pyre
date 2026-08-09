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


# A Windows FILETIME counts 100ns ticks, so a nanosecond that is not a multiple
# of 100 is not a time that filesystem can hold; ext4 and APFS both store the
# nanosecond itself. The negative range is not what differs — only how finely
# it is kept — so the checks below round to what the host can hold rather than
# skipping a platform.
GRANULARITY_NS = 100 if sys.platform == "win32" else 1


def storable(ns):
    """`ns` as the filesystem under this test reads it back."""
    return ns // GRANULARITY_NS * GRANULARITY_NS


d = tempfile.mkdtemp()
atexit.register(shutil.rmtree, d, ignore_errors=True)
p = os.path.join(d, "f")
with open(p, "wb") as f:
    f.write(b"0123456789")

# ── utime before the epoch ────────────────────────────────────────────────
# The seconds are the floor of the value and the nanoseconds are what is left
# above it, so -1ns is the last nanosecond of 1969 rather than a value with no
# representation.
os.utime(p, ns=(-1, -1))
check(
    os.stat(p).st_mtime_ns == storable(-1),
    f"utime(ns=(-1,-1)) -> {os.stat(p).st_mtime_ns}",
)
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
# before the keyword-only marker.
os.utime(p, times=(-5.0, -6.0))
check(os.stat(p).st_mtime_ns == -6_000_000_000, "utime(times=...) by keyword")
raises(lambda: os.utime(p, (1, 2), times=(3, 4)), TypeError)

# ── what a second may be spelled as ───────────────────────────────────────
# A value that is not a number is refused by type rather than by a failed
# conversion, one no `time_t` can hold by overflow rather than by value, and
# a NaN has no floor to take at all.
e = raises(lambda: os.utime(p, ("a", "b")), TypeError)
check(str(e) == "argument must be int or float, not str", str(e))
e = raises(lambda: os.utime(p, (None, 1)), TypeError)
check(str(e) == "argument must be int or float, not NoneType", str(e))
for far in (1e30, float("inf"), 2**200, -(2**200)):
    e = raises(lambda: os.utime(p, (far, 0)), OverflowError)
    check(str(e) == "timestamp out of range for platform time_t", f"{far!r}: {e}")
e = raises(lambda: os.utime(p, (float("nan"), 0.0)), ValueError)
check(str(e) == "Invalid value NaN (not a number)", str(e))
# `times` is the argument that also has a `None` spelling, and its message
# names it; `ns` has no such form.
e = raises(lambda: os.utime(p, (1,)), TypeError)
check(str(e) == "utime: 'times' must be either a tuple of two ints or None", str(e))
e = raises(lambda: os.utime(p, ns=(1,)), TypeError)
check(str(e) == "utime: 'ns' must be a tuple of two ints", str(e))

# `ns` is split with divmod before anything is narrowed, so a nanosecond count
# too wide for a `time_t` is only refused when the SECOND it names is.
os.utime(p, ns=(2**62, 2**62))
check(os.stat(p).st_mtime_ns == storable(2**62), os.stat(p).st_mtime_ns)

# Nanoseconds run out of an `int64` in 2262, and a file may carry a later time
# than that — `st_mtime_ns` is the number it is rather than a wrap.
#
# Only a filesystem that reaches such a date can be asked about it, and by
# construction that cannot be every one: an APFS timestamp IS an int64 of
# nanoseconds, so 2262 is its ceiling too, and ext4 stores a 34-bit second and
# stops in 2446. Both clamp what they are handed (some hosts refuse it), where
# NTFS counts 100ns ticks to the year 30828. A clamp stops at the last instant
# the store can name, and that need not fall on a whole second: APFS stops at
# `2**63 - 1` nanoseconds, whose remainder is 854775807ns. So what every
# platform is held to is that the count neither wrapped nor names a later
# instant than the one asked for, and that the two fields read back the same
# time; the exact value only where the second survived the round trip.
FAR = 8.8e11
try:
    os.utime(p, (FAR, FAR))
except OSError:
    pass
else:
    st = os.stat(p)
    check(0 < st.st_mtime_ns <= int(FAR) * 1_000_000_000, st.st_mtime_ns)
    second, nanosecond = divmod(st.st_mtime_ns, 1_000_000_000)
    check(st.st_mtime == second + 1e-9 * nanosecond, (st.st_mtime, st.st_mtime_ns))
    if st.st_mtime == FAR:
        check(st.st_mtime_ns == 880_000_000_000_000_000_000, st.st_mtime_ns)

# Back to a time the rest of the file can be reasoned about.
os.utime(p, ns=(1_000_000_000, 2_000_000_000))

# ── pathconf's indeterminate limit ────────────────────────────────────────
# Every name the table carries either answers with a number or refuses the
# question — the terminal-only limits are not ones a regular file has. What no
# answer may be is None: a host with no determinate value says so with -1.
#
# `pathconf` and the `pathconf_names` table it resolves through are a POSIX
# surface; neither runtime carries them on Windows, so there is nothing to
# compare there. `hasattr` rather than a platform list: what the section needs
# is the name, not the platform.
def limits(target):
    for name in sorted(os.pathconf_names):
        try:
            limit = os.pathconf(target, name)
        except OSError:
            continue
        check(isinstance(limit, int), f"pathconf({name!r}) answered {limit!r}")
        check(limit >= -1, f"pathconf({name!r}) answered {limit}")
        yield name, limit


check(
    hasattr(os, "pathconf") == hasattr(os, "pathconf_names"),
    "one of pathconf / pathconf_names is here without the other",
)
if hasattr(os, "pathconf"):
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
