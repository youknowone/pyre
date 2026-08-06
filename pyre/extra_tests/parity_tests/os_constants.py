"""The posix constant families carry the host's values, or are not there.

A constant bound to zero is not a placeholder a caller can detect: `os._exit`
takes an EX_* straight to the exit status, `statvfs(...).f_flag` is masked with
the ST_* bits, and the RTLD_* set is handed back to `dlopen`, where zero asks
for `RTLD_LOCAL | RTLD_LAZY` whatever was named. Each family is checked here
for the shape it has to have — the members distinct, the flag members single
bits — because binding them all to one value is what collapses that shape.

The exact numbering is the host's own and is not asserted, except for the EX_*
family, whose header is a verbatim descendant of the 4.3BSD one everywhere.
"""

import os
import sys


def check(cond, what):
    if not cond:
        raise AssertionError(what)


SYSEXITS = [
    ("EX_OK", 0),
    ("EX_USAGE", 64),
    ("EX_DATAERR", 65),
    ("EX_NOINPUT", 66),
    ("EX_NOUSER", 67),
    ("EX_NOHOST", 68),
    ("EX_UNAVAILABLE", 69),
    ("EX_SOFTWARE", 70),
    ("EX_OSERR", 71),
    ("EX_OSFILE", 72),
    ("EX_CANTCREAT", 73),
    ("EX_IOERR", 74),
    ("EX_TEMPFAIL", 75),
    ("EX_PROTOCOL", 76),
    ("EX_NOPERM", 77),
    ("EX_CONFIG", 78),
]
DLOPEN = ["RTLD_LAZY", "RTLD_NOW", "RTLD_GLOBAL", "RTLD_LOCAL", "RTLD_NODELETE", "RTLD_NOLOAD"]
STATVFS = ["ST_RDONLY", "ST_NOSUID"]
SCHED = ["SCHED_OTHER", "SCHED_FIFO", "SCHED_RR"]
FAMILIES = [name for name, _ in SYSEXITS] + DLOPEN + STATVFS + SCHED

if sys.platform == "win32":
    # None of the four headers is one Windows carries.
    for name in FAMILIES + ["SCHED_BATCH", "SCHED_IDLE", "RTLD_DEEPBIND"]:
        check(not hasattr(os, name), f"windows grew an os.{name}")
    print("OK")
    raise SystemExit

for name in FAMILIES:
    check(hasattr(os, name), f"no os.{name}")
    check(isinstance(getattr(os, name), int), f"os.{name} is not a number")

# ── <sysexits.h> ──────────────────────────────────────────────────────────
for name, value in SYSEXITS:
    check(getattr(os, name) == value, f"os.{name} is {getattr(os, name)}, not {value}")

# ── <dlfcn.h> ─────────────────────────────────────────────────────────────
# Six modes that are or'd together, so each is a single bit and no two name
# the same one. RTLD_LOCAL is the absence of RTLD_GLOBAL on some hosts and so
# is legitimately zero; the rest are not.
flags = [getattr(os, name) for name in DLOPEN]
check(len(set(flags)) == len(flags), f"the RTLD_* set collapses: {dict(zip(DLOPEN, flags))}")
for name, value in zip(DLOPEN, flags):
    check(value & (value - 1) == 0, f"os.{name} is {value:#x}, which is not one bit")
    check(value != 0 or name == "RTLD_LOCAL", f"os.{name} is zero")
check(os.RTLD_LAZY & os.RTLD_NOW == 0, "RTLD_LAZY and RTLD_NOW share a bit")

# RTLD_DEEPBIND is glibc's own extension rather than anything the header has.
if not sys.platform.startswith("linux"):
    check(not hasattr(os, "RTLD_DEEPBIND"), "RTLD_DEEPBIND outside glibc")

# ── statvfs f_flag ────────────────────────────────────────────────────────
bits = [getattr(os, name) for name in STATVFS]
check(len(set(bits)) == len(bits), f"the ST_* set collapses: {dict(zip(STATVFS, bits))}")
for name, value in zip(STATVFS, bits):
    check(value != 0, f"os.{name} is zero")
    check(value & (value - 1) == 0, f"os.{name} is {value:#x}, which is not one bit")

if hasattr(os, "statvfs"):
    # The flag word is what the bits exist to read, and a directory the test
    # just wrote into is not on a read-only filesystem.
    flag = os.statvfs(os.getcwd()).f_flag
    check(isinstance(flag, int), f"f_flag is {flag!r}")
    check(not flag & os.ST_RDONLY, f"the cwd reads as read-only: f_flag={flag:#x}")

# ── <sched.h> ─────────────────────────────────────────────────────────────
# Policy numbers rather than flags, so they are distinct but not bits, and
# their numbering disagrees between Linux, the BSDs and Darwin.
policies = [getattr(os, name) for name in SCHED]
check(len(set(policies)) == len(policies), f"the SCHED_* set collapses: {dict(zip(SCHED, policies))}")

# SCHED_BATCH and SCHED_IDLE are Linux's own additions, and arrive together.
check(
    hasattr(os, "SCHED_BATCH") == hasattr(os, "SCHED_IDLE"),
    "SCHED_BATCH and SCHED_IDLE disagree about being here",
)
if not sys.platform.startswith("linux"):
    check(not hasattr(os, "SCHED_BATCH"), "SCHED_BATCH outside Linux")
else:
    check(len({os.SCHED_BATCH, os.SCHED_IDLE} | set(policies)) == 5, "SCHED_BATCH/IDLE collide")

print("OK")
