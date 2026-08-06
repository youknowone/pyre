"""What the extension module publishes, and what it leaves to os.py.

`os.py` star-imports the extension module and then writes a handful of names
itself. Three things go wrong when the module binds one of those anyway: the
name is counted twice in `os.__all__` (os.py lists its own in `__all__` too),
the module's version is a stub until os.py's definition replaces it, and where
os.py guards its definition on the name being free — the spawn family at
os.py:881 — it never runs at all.

The other half is the C entry points. `openat`, `faccessat` and the rest are
how the calls above them are served, not calls of their own, and a name bound
for one is a capability a caller can probe for and believe.
"""

import os
import sys
from collections import Counter


def check(cond, what):
    if not cond:
        raise AssertionError(what)


# ── os.__all__ names each export once ─────────────────────────────────────
# A name the extension module binds arrives through `_get_exports_list`, and
# a name os.py writes arrives through its own `__all__` — so a name in both
# places is listed twice. That is the tell, and it is cheap to check for all
# of them at once.
twice = sorted(name for name, count in Counter(os.__all__).items() if count > 1)
check(not twice, f"os.__all__ lists these more than once: {twice}")

# ── the names os.py owns ──────────────────────────────────────────────────
for name in ("popen", "get_exec_path", "getenv", "fdopen", "fsencode", "fsdecode"):
    fn = getattr(os, name)
    check(type(fn).__name__ == "function", f"os.{name} is {type(fn).__name__}, not os.py's")

check(os.SEEK_SET == 0 and os.SEEK_CUR == 1 and os.SEEK_END == 2, "the SEEK_* trio moved")
check(isinstance(os.get_exec_path(), list), "get_exec_path answers no list")
check(os.getenv("PATH", "") == os.environ.get("PATH", ""), "getenv and environ disagree")

if sys.platform != "win32":
    with os.popen("echo parity") as pipe:
        check(pipe.read() == "parity\n", "popen read nothing back")

# ── the C entry points are not names ──────────────────────────────────────
# `fstatat`/`faccessat`/`futimens`/`futimes`/`fdopendir` are how `dir_fd` and
# a descriptor path are served; `setenv` is the C spelling of `putenv`.
for name in ("fstatat", "faccessat", "futimens", "futimes", "fdopendir", "setenv"):
    check(not hasattr(os, name), f"os.{name} is a C entry point, not an export")

# `pipe2`, `dup3` and the scheduling-policy calls are Linux's own. Their
# presence there is not asserted — this file also runs against a build that
# does not serve them yet, and an absent name is what that build should say.
if not sys.platform.startswith("linux"):
    for name in (
        "pipe2",
        "dup3",
        "sched_getparam",
        "sched_setparam",
        "sched_getscheduler",
        "sched_setscheduler",
    ):
        check(not hasattr(os, name), f"os.{name} outside Linux")

# ── waitid's option flags are numbers ─────────────────────────────────────
# A call answering None is neither the number nor a name a caller can tell
# apart from one. The values are the host's; that they are distinct single
# bits is not.
if sys.platform != "win32":
    options = {name: getattr(os, name) for name in ("WEXITED", "WSTOPPED", "WNOWAIT")}
    for name, value in options.items():
        check(isinstance(value, int), f"os.{name} is {type(value).__name__}, not a number")
        check(value != 0, f"os.{name} is zero")
        check(value & (value - 1) == 0, f"os.{name} is {value:#x}, which is not one bit")
    check(len(set(options.values())) == 3, f"the waitid options collapse: {options}")

print("OK")
