"""A structseq type is built once, not rebuilt for each answer.

`PyStructSequence_InitType2` runs at module init and the type it leaves is what
every later result is an instance of, so `type(os.stat(a)) is type(os.stat(b))`
and an `isinstance` check written against one answer holds for the next. A
constructor that builds its type inside the call instead returns a fresh class
each time: the values compare equal, the types never do, and the failure only
shows up in code that kept a type from an earlier call.

Each name is asked for twice and skipped when the platform does not carry it —
`statvfs`/`uname` are POSIX, `getwindowsversion` is Windows, and a terminal size
is not a question a redirected stdout can answer.
"""

import os
import sys
import time


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def answers():
    """Every structseq this platform can be asked for, as a name and a thunk."""
    yield "os.stat_result", lambda: os.stat(sys.executable)
    yield "os.times_result", os.times
    yield "time.struct_time", time.localtime
    yield "sys.version_info", lambda: sys.version_info
    yield "sys.flags", lambda: sys.flags
    yield "os.terminal_size", os.get_terminal_size
    if hasattr(os, "statvfs"):
        yield "os.statvfs_result", lambda: os.statvfs(".")
    if hasattr(os, "uname"):
        yield "os.uname_result", os.uname
    if hasattr(sys, "getwindowsversion"):
        yield "sys.getwindowsversion", sys.getwindowsversion


asked = 0
for name, call in answers():
    try:
        first = call()
    except (AttributeError, OSError):
        # Not a question this host can answer; nothing to compare.
        continue
    second = call()
    asked += 1
    check(
        type(first) is type(second),
        f"{name} built a second type for its second answer",
    )
    # The type is a tuple subclass, which is what makes the sequence half of a
    # structseq work at all — and what a fresh-per-call type would still get
    # right, so it is checked beside the identity rather than instead of it.
    check(isinstance(first, tuple), f"{name} is not a tuple subclass")
    check(
        type(first).__name__ == type(second).__name__,
        f"{name} disagreed with itself about its own name",
    )

check(asked >= 5, f"only {asked} structseq(s) were reachable to compare")

print("OK")
